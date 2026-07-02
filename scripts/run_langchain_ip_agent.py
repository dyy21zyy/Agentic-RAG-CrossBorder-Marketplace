#!/usr/bin/env python
"""Run the Phase 5 single LangChain IP agent."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crossborder_agentic_rag.agents.langchain_agent import build_langchain_ip_agent
from crossborder_agentic_rag.ingestion.io_utils import read_chunks_jsonl
from crossborder_agentic_rag.llm.embeddings import build_embedding_provider
from crossborder_agentic_rag.retrieval import HybridRetriever, LocalBM25Retriever, SourceBalancedRetriever, build_reranker
from crossborder_agentic_rag.storage.duckdb_store import DuckDBStore
from crossborder_agentic_rag.storage.milvus_store import MilvusChunkStore
from crossborder_agentic_rag.tools.ip_tools import build_ip_tools


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query", required=True)
    p.add_argument("--chunks-path", default="data/processed/chunks_qa_300k.jsonl")
    p.add_argument("--duckdb-path", default="data/processed/ip_lookup.duckdb")
    p.add_argument("--use-milvus", action="store_true")
    p.add_argument("--collection-name", default="ip_rag_collection")
    p.add_argument("--embedding-provider", default=os.getenv("EMBEDDING_PROVIDER", "local"))
    p.add_argument("--retrieval-mode", choices=["bm25_only", "dense_only", "hybrid_rrf", "hybrid_rerank", "source_balanced"], default="hybrid_rerank")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--candidate-k", type=int, default=50)
    p.add_argument("--llm-provider", default=os.getenv("LLM_PROVIDER", "template"))
    p.add_argument("--llm-model", default=os.getenv("LLM_MODEL"))
    p.add_argument("--output-json", action="store_true")
    p.add_argument("--output")
    return p.parse_args()


def _milvus_uri() -> str | None:
    return os.getenv("RAG_MILVUS_URI") or os.getenv("MILVUS_URI")


def build_retriever(args: argparse.Namespace, errors: list[str]) -> tuple[Any, Any | None]:
    chunks = read_chunks_jsonl(args.chunks_path)
    bm25 = LocalBM25Retriever(chunks)
    provider = None
    store = None
    mode = args.retrieval_mode
    needs_dense = args.use_milvus or mode in {"dense_only", "hybrid_rrf", "hybrid_rerank", "source_balanced"}
    if needs_dense:
        try:
            provider = build_embedding_provider(args.embedding_provider)
            vec = provider.embed_query(args.query)
            uri = _milvus_uri()
            if not uri:
                raise ValueError("RAG_MILVUS_URI or MILVUS_URI is required for dense/hybrid retrieval")
            store = MilvusChunkStore(uri, os.getenv("MILVUS_TOKEN"), args.collection_name, len(vec))
        except Exception as exc:
            errors.append(f"Dense retrieval unavailable, falling back to BM25: {exc}")
            mode = "bm25_only"
    reranker = None
    if mode == "hybrid_rerank":
        try:
            reranker = build_reranker(os.getenv("RERANKER_PROVIDER", "noop"), os.getenv("RERANKER_MODEL"))
        except Exception as exc:
            errors.append(f"Reranker unavailable, falling back to hybrid_rrf: {exc}")
            mode = "hybrid_rrf"
    base = HybridRetriever(provider, bm25, store, reranker)
    if args.retrieval_mode == "source_balanced":
        return SourceBalancedRetriever(base, mode=mode, per_source_k=args.top_k, final_k=args.top_k, candidate_k=args.candidate_k), provider
    return _ConfiguredRetriever(base, mode, args.top_k, args.candidate_k), provider


class _ConfiguredRetriever:
    def __init__(self, base: Any, mode: str, top_k: int, candidate_k: int) -> None:
        self.base = base
        self.mode = mode
        self.top_k = top_k
        self.candidate_k = candidate_k

    def retrieve(self, query: str, source_types: list[str] | None = None, top_k: int | None = None, candidate_k: int | None = None, **kwargs: Any) -> list[Any]:
        return self.base.retrieve(query, source_types=source_types, top_k=top_k or self.top_k, candidate_k=candidate_k or self.candidate_k, mode=self.mode, **kwargs)


def build_duckdb(path: str, errors: list[str]) -> DuckDBStore | None:
    if not path:
        return None
    try:
        store = DuckDBStore(path)
        store.initialize_schema()
        return store
    except Exception as exc:
        errors.append(f"DuckDB unavailable: {exc}")
        return None


def build_llm(args: argparse.Namespace):
    provider = (args.llm_provider or "template").lower().replace("-", "_")
    if provider in {"openai", "openai_compatible"}:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("Install langchain-openai to use --llm-provider openai/openai_compatible") from exc
        return ChatOpenAI(model=args.llm_model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"), temperature=0)
    try:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
    except ImportError as exc:
        raise ImportError("Install langchain-core for template smoke runs") from exc
    return FakeListChatModel(responses=["Evidence was retrieved by the configured IP tools. Review intermediate_steps for citations and missing evidence."])


def _manual_tool_run(query: str, tools: list[Any]) -> tuple[str, list[Any]]:
    selected = [t for t in tools if t.name in {"trademark_search_tool", "patent_search_tool", "litigation_search_tool", "graph_rag_tool"}]
    steps = []
    for tool in selected:
        content = tool.invoke(query)
        steps.append({"tool": tool.name, "output": content})
    return "Evidence was retrieved by trademark, patent, litigation, and optional GraphRAG tools. Use returned evidence only; missing evidence is shown in tool outputs.", steps


def _collect(parsed_steps: list[Any]) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    citations: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for step in parsed_steps:
        if isinstance(step, tuple) and len(step) == 2:
            action, output = step
            name = getattr(action, "tool", None)
        else:
            name = step.get("tool") if isinstance(step, dict) else None
            output = step.get("output") if isinstance(step, dict) else step
        tool_calls.append({"tool": name, "output": output})
        try:
            payload = json.loads(output) if isinstance(output, str) else output
        except Exception:
            continue
        for item in payload.get("evidence", []) if isinstance(payload, dict) else []:
            evidence.append(item)
            cid = item.get("chunk_id") or item.get("doc_id")
            if cid:
                citations.append(str(cid))
    return evidence, sorted(set(citations)), tool_calls


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    errors: list[str] = []
    answer = ""
    steps: list[Any] = []
    try:
        retriever, embedding_provider = build_retriever(args, errors)
        duckdb_store = build_duckdb(args.duckdb_path, errors)
        graph_retriever = None
        tools = build_ip_tools(retriever, embedding_provider=embedding_provider, duckdb_store=duckdb_store, graph_retriever=graph_retriever, default_top_k=args.top_k, candidate_k=args.candidate_k)
        try:
            llm = build_llm(args)
            executor = build_langchain_ip_agent(llm, tools)
            result = executor.invoke({"input": args.query})
            answer = result.get("output", "")
            steps = result.get("intermediate_steps", [])
        except Exception as exc:
            errors.append(f"LangChain AgentExecutor unavailable, used deterministic tool fallback: {exc}")
            answer, steps = _manual_tool_run(args.query, tools)
    except Exception as exc:
        errors.append(str(exc))
    evidence, citations, tool_calls = _collect(steps)
    output = {"query": args.query, "answer": answer, "tool_calls": tool_calls, "intermediate_steps": tool_calls, "evidence": evidence, "citations": citations, "latency_ms": int((time.perf_counter() - started) * 1000), "errors": errors}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text if args.output_json else output["answer"])
    return 0 if answer else 1


if __name__ == "__main__":
    raise SystemExit(main())
