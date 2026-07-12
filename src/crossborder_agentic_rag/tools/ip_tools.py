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


def build_ip_tools(
    retriever: Any,
    embedding_provider: Any = None,
    duckdb_store: Any = None,
    graph_retriever: Any = None,
    default_top_k: int = 8,
    candidate_k: int = 50,
    dense_k: int | None = None,
    bm25_k: int | None = None,
    rrf_k: int | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Build LangChain tool objects without relying on module-level state."""

    # Formal retrieval K settings:
    # Dense Top20, BM25 Top20, RRF Top10, Reranker Top5.
    if retriever is not None:
        for _name, _value in [
            ("dense_k", dense_k if dense_k is not None else candidate_k),
            ("bm25_k", bm25_k if bm25_k is not None else candidate_k),
            ("rrf_k_final", rrf_k if rrf_k is not None else 10),
            ("candidate_k", candidate_k),
            ("top_k", default_top_k),
        ]:
            try:
                setattr(retriever, _name, _value)
            except Exception:
                pass
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
            result = graph_retriever.retrieve(query)
            if isinstance(result, dict):
                return _json({"tool_name": "graph_rag_tool", "available": True, "evidence": [], **result})
            return _json({"tool_name": "graph_rag_tool", "available": True, "evidence": [_chunk_to_dict(h) for h in result]})
        except Exception as exc:
            return _json({"tool_name": "graph_rag_tool", "available": False, "evidence": [], "error": str(exc)})

    return [
        Tool.from_function(
            source_tool("trademark_search_tool", "trademark"),
            name="trademark_search_tool",
            description=(
                "Use this tool for trademark-related evidence retrieval. "
                "Best for questions about word marks, brand names, logos, trademark similarity, "
                "counterfeit risk, unauthorized brand use, Nice classes, goods/services, "
                "registration records, serial numbers, and trademark ownership. "
                "Input should be a natural-language search query. "
                "Output contains trademark evidence chunks with chunk_id, doc_id, source_type, "
                "metadata, content, and retrieval score."
            ),
        ),
        Tool.from_function(
            source_tool("patent_search_tool", "patent"),
            name="patent_search_tool",
            description=(
                "Use this tool for patent-related evidence retrieval. "
                "Best for questions about patent claims, product technical features, inventions, "
                "utility patents, design patents, claim infringement, functional similarity, "
                "and whether a product may overlap with existing US patent claims. "
                "Input should describe the product, feature, or patent concern. "
                "Output contains patent evidence chunks with chunk_id, doc_id, source_type, "
                "metadata, content, and retrieval score."
            ),
        ),
        Tool.from_function(
            source_tool("litigation_search_tool", "litigation"),
            name="litigation_search_tool",
            description=(
                "Use this tool for patent or trademark litigation evidence retrieval. "
                "Best for questions about lawsuits, docket records, cases, plaintiffs, defendants, "
                "parties, asserted patents, case numbers, litigation history, companies being sued, "
                "or whether a product/company/brand is connected to IP disputes. "
                "Input should be a natural-language litigation or case search query. "
                "Output contains litigation evidence chunks with chunk_id, doc_id, source_type, "
                "metadata, content, and retrieval score."
            ),
        ),
        Tool.from_function(
            duckdb_run,
            name="duckdb_lookup_tool",
            description=(
                "Use this tool for exact structured lookup in DuckDB. "
                "Best when the query contains exact identifiers or fields such as registration number, "
                "serial number, word mark, patent id, patent number, case number, Nice class, "
                "goods/services, party name, or other structured metadata. "
                "Prefer this tool over semantic retrieval when the user asks for exact field values, "
                "exact records, or identifier-based lookup. "
                "Output contains structured SQL lookup results and missing-field information."
            ),
        ),
        Tool.from_function(
            graph_run,
            name="graph_rag_tool",
            description=(
                "Use this tool for GraphRAG entity-relation expansion. "
                "Best for multi-hop or relationship questions such as company to trademark, "
                "company to litigation case, case to patent, party to asserted patent, "
                "trademark to owner, patent to claim, or entity-neighborhood exploration. "
                "Use this tool when relevant evidence may be connected through entities rather than "
                "direct lexical or semantic text matches. "
                "Output contains graph-expanded evidence or relationship information when available."
            ),
        ),
    ]
