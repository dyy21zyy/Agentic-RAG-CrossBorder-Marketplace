"""LangChain tools for Phase 5 intellectual-property retrieval."""
from __future__ import annotations

import json
import re
from typing import Any, Callable


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": getattr(chunk, "chunk_id", None),
        "doc_id": getattr(chunk, "doc_id", None),
        "source_type": getattr(chunk, "source_type", None),
        "source_subtype": getattr(chunk, "source_subtype", None),
        "title": getattr(chunk, "title", None),
        "content": getattr(chunk, "content", None),
        "metadata": getattr(chunk, "metadata", {}) or {},
        "score": getattr(chunk, "score", None),
    }


def _retrieve(retriever: Any, query: str, source_type: str, top_k: int, candidate_k: int) -> tuple[list[dict[str, Any]], str | None]:
    try:
        try:
            hits = retriever.retrieve(query, source_types=[source_type], top_k=top_k, candidate_k=candidate_k)
        except TypeError:
            hits = retriever.retrieve(query, source_types=[source_type])
        return [_chunk_to_dict(hit) for hit in hits], None
    except Exception as exc:  # tools must not crash the agent
        return [], str(exc)


def _extract_registration_number(query: str) -> str | None:
    patterns = [r"registration\s*(?:number|no\.?|#)?\s*[:#]?\s*([A-Z0-9-]{4,})", r"\breg\.?\s*(?:no\.?|#)?\s*[:#]?\s*([A-Z0-9-]{4,})"]
    return _first_match(query, patterns)


def _extract_patent_id(query: str) -> str | None:
    patterns = [r"\b(?:US)?\s*patent\s*(?:number|no\.?|id|#)?\s*[:#]?\s*(US?\s*[0-9][0-9A-Z,/-]{3,})", r"\b(US\s*[0-9][0-9A-Z,/-]{3,})\b"]
    value = _first_match(query, patterns)
    return value.replace(" ", "") if value else None


def _extract_case_number(query: str) -> str | None:
    patterns = [r"case\s*(?:number|no\.?|#)?\s*[:#]?\s*([0-9][0-9A-Za-z:._-]{3,})", r"\b([0-9]{1,2}:[0-9]{2}-[a-z]{2}-[0-9]{3,})\b"]
    return _first_match(query, patterns)


def _extract_quoted(query: str) -> str | None:
    m = re.search(r"[\"']([^\"']{2,80})[\"']", query)
    return m.group(1).strip() if m else None


def _extract_word_mark(query: str) -> str | None:
    explicit = re.search(r"word\s*mark\s*[:=]?\s*([A-Za-z0-9][A-Za-z0-9 &'-]{1,60})", query, re.I)
    if explicit:
        return explicit.group(1).strip(" .")
    return _extract_quoted(query)


def _first_match(query: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        m = re.search(pattern, query, re.I)
        if m:
            return m.group(1).strip()
    return None


def _safe_call(obj: Any, method_name: str, *args: Any) -> dict[str, Any]:
    if obj is None:
        return {"supported": False, "message": "DuckDBStore is not configured."}
    method = getattr(obj, method_name, None)
    if method is None:
        return {"supported": False, "message": f"DuckDBStore does not support {method_name}."}
    try:
        return {"supported": True, "rows": method(*args)}
    except Exception as exc:
        return {"supported": True, "error": str(exc), "rows": []}


def _duckdb_lookup(duckdb_store: Any, query: str) -> dict[str, Any]:
    lookups: dict[str, Any] = {}
    if reg := _extract_registration_number(query):
        lookups["registration_number"] = _safe_call(duckdb_store, "lookup_trademark_by_registration_number", reg)
    if mark := _extract_word_mark(query):
        lookups["word_mark"] = {
            "trademarks": _safe_call(duckdb_store, "lookup_trademark_by_word_mark", mark),
            "classes": _safe_call(duckdb_store, "lookup_trademark_classes_by_word_mark", mark),
            "goods_services": _safe_call(duckdb_store, "lookup_trademark_goods_services_by_word_mark", mark),
        }
    if patent := _extract_patent_id(query):
        lookups["patent_id"] = _safe_call(duckdb_store, "lookup_patent_by_id", patent)
        lookups["litigation_by_patent"] = _safe_call(duckdb_store, "lookup_litigation_by_patent", patent)
    if case := _extract_case_number(query):
        lookups["case_number"] = {
            "case": _safe_call(duckdb_store, "lookup_litigation_by_case", case),
            "parties": _safe_call(duckdb_store, "lookup_litigation_parties_by_case", case),
            "documents": _safe_call(duckdb_store, "lookup_litigation_documents_by_case", case),
            "patents": _safe_call(duckdb_store, "lookup_litigation_patents_by_case", case),
        }
    if not lookups:
        lookups["missing"] = ["registration_number", "word_mark", "patent_id/patent_number", "case_number"]
    return lookups


def build_ip_tools(retriever: Any, embedding_provider: Any = None, duckdb_store: Any = None, graph_retriever: Any = None, default_top_k: int = 8, candidate_k: int = 50) -> list[Any]:
    """Build LangChain tool objects without relying on module-level state."""
    try:
        from langchain_core.tools import Tool
    except ImportError:
        class Tool:  # minimal CLI smoke-run fallback when LangChain is not installed
            def __init__(self, func, name: str, description: str) -> None:
                self.func = func
                self.name = name
                self.description = description

            @classmethod
            def from_function(cls, func, name: str, description: str):
                return cls(func, name, description)

            def invoke(self, query):
                return self.func(query)

    def source_tool(name: str, source_type: str) -> Callable[[str], str]:
        def run(query: str) -> str:
            evidence, error = _retrieve(retriever, query, source_type, default_top_k, candidate_k)
            payload = {"tool_name": name, "source_type": source_type, "evidence": evidence, "sql_results": {}, "missing": [], "error": error}
            if duckdb_store is not None:
                payload["sql_results"] = _duckdb_lookup(duckdb_store, query)
            elif any([_extract_registration_number(query), _extract_word_mark(query), _extract_patent_id(query), _extract_case_number(query)]):
                payload["missing"].append("duckdb_store")
            return _json(payload)
        return run

    def duckdb_run(query: str) -> str:
        return _json({"tool_name": "duckdb_lookup_tool", "sql_results": _duckdb_lookup(duckdb_store, query), "error": None})

    def graph_run(query: str) -> str:
        if graph_retriever is None:
            return _json({"tool_name": "graph_rag_tool", "available": False, "evidence": [], "message": "GraphRAG index not available"})
        try:
            hits = graph_retriever.retrieve(query)
            return _json({"tool_name": "graph_rag_tool", "available": True, "evidence": [_chunk_to_dict(h) for h in hits]})
        except Exception as exc:
            return _json({"tool_name": "graph_rag_tool", "available": False, "evidence": [], "error": str(exc)})

    return [
        Tool.from_function(source_tool("trademark_search_tool", "trademark"), name="trademark_search_tool", description="Search only trademark evidence for the query."),
        Tool.from_function(source_tool("patent_search_tool", "patent"), name="patent_search_tool", description="Search only patent evidence for the query."),
        Tool.from_function(source_tool("litigation_search_tool", "litigation"), name="litigation_search_tool", description="Search only litigation evidence for the query."),
        Tool.from_function(duckdb_run, name="duckdb_lookup_tool", description="Run structured exact DuckDB lookups for registration number, word mark, patent id/number, or case number."),
        Tool.from_function(graph_run, name="graph_rag_tool", description="Search GraphRAG relationships when the graph index is available."),
    ]
