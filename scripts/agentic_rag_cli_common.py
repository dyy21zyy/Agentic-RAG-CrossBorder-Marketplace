"""Shared helpers for agentic and baseline RAG CLIs."""
from __future__ import annotations

import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from crossborder_agentic_rag.agents.answer import synthesize_answer
from crossborder_agentic_rag.agents.graph import AgenticRAG
from crossborder_agentic_rag.ingestion.io_utils import read_chunks_jsonl
from crossborder_agentic_rag.llm.embeddings import FakeEmbeddingProvider, build_embedding_provider
from crossborder_agentic_rag.retrieval import HybridRetriever, LocalBM25Retriever, build_reranker
from crossborder_agentic_rag.retrieval.utils import dedupe_chunks
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
from crossborder_agentic_rag.schemas.queries import QueryPlan
from crossborder_agentic_rag.storage.duckdb_store import DuckDBStore
from crossborder_agentic_rag.storage.milvus_store import MilvusChunkStore

DEFAULT_SOURCE_TYPES = ["trademark", "patent", "litigation"]


def load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def demo_chunks() -> list[EvidenceChunk]:
    return [
        EvidenceChunk("tm:mercedes:0", "tm-mercedes", "trademark", "classes", "MERCEDES trademark", "MERCEDES registration covers Nice class 12 vehicles and related goods.", {"chunk_index": 0}),
        EvidenceChunk("pat:bag:0", "pat-smart-bag", "patent", "claims", "Smart bag patent", "A smart travel bag patent claim describes location tracking and locking controls.", {"chunk_index": 0}),
        EvidenceChunk("lit:case:0", "lit-case", "litigation", "docket", "Marketplace IP case", "A marketplace case discusses seller infringement allegations and platform takedown evidence.", {"chunk_index": 0}),
        EvidenceChunk("policy:temu-ip:0", "temu-ip", "policy", "policy_enforcement", "Temu IP Policy", "Policy prohibits counterfeit goods and trademark infringement.", {"chunk_index": 0}),
    ]


def parse_source_types(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_SOURCE_TYPES)
    return [item.strip() for item in value.split(",") if item.strip()]


def compact_evidence(chunk: EvidenceChunk) -> dict[str, Any]:
    data = chunk.to_dict()
    return {
        "chunk_id": data["chunk_id"],
        "doc_id": data["doc_id"],
        "source_type": data["source_type"],
        "source_subtype": data["source_subtype"],
        "title": data["title"],
        "score": data["score"],
        "metadata": data["metadata"],
        "content_preview": data["content"][:240],
    }


def evidence_manifest(evidence: list[EvidenceChunk]) -> list[dict[str, Any]]:
    return [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "source_type": c.source_type, "title": c.title} for c in evidence]


def generate_grounded_answer(query: str, deterministic_answer: str, evidence: list[EvidenceChunk], provider: str | None, model: str | None) -> tuple[str | None, str | None]:
    """Generate an optional grounded answer.

    This dependency-free default keeps CI offline-safe while giving both pipeline
    modes one shared function to patch or replace in later experiments.
    """
    if not provider or provider in {"template", "none"}:
        titles = "; ".join(c.title for c in evidence[:3]) or "no evidence"
        return f"{deterministic_answer}\n\nGrounded summary for '{query}' using: {titles}", None
    return None, f"LLM provider '{provider}' is not configured in this offline CLI."


class DemoVectorStore:
    def __init__(self, chunks: list[EvidenceChunk]) -> None:
        self.chunks = chunks

    def dense_search(self, dense_vector, filters=None, top_k: int = 20):
        return self.chunks[:top_k]


def build_runtime(args: Namespace) -> tuple[Any | None, Any, Any | None]:
    chunks: list[EvidenceChunk] = []
    embedding = None
    if getattr(args, "demo", False):
        chunks = demo_chunks()
        embedding = FakeEmbeddingProvider()
        reranker = build_reranker(args.reranker_provider, args.reranker_model) if args.retrieval_mode == "hybrid_rerank" else None
        vector_store = DemoVectorStore(chunks) if args.retrieval_mode in {"dense_only", "hybrid_rrf", "hybrid_rerank"} else None
        return None, HybridRetriever(embedding, LocalBM25Retriever(chunks), vector_store, reranker), embedding
    if getattr(args, "chunks_path", None):
        path = Path(args.chunks_path)
        if not path.is_file():
            raise FileNotFoundError(f"Chunks path does not exist: {path}")
        chunks = read_chunks_jsonl(path)
        embedding = build_embedding_provider(args.embedding_provider)
        reranker = build_reranker(args.reranker_provider, args.reranker_model) if args.retrieval_mode == "hybrid_rerank" else None
        return None, HybridRetriever(embedding, LocalBM25Retriever(chunks), None, reranker), embedding
    if getattr(args, "use_milvus", False):
        embedding = build_embedding_provider(args.embedding_provider)
        dim = len(embedding.embed_query("dimension probe"))
        store = MilvusChunkStore(os.getenv("MILVUS_URI", "http://localhost:19530"), os.getenv("MILVUS_TOKEN"), args.collection_name, dim)
        store.connect(); store.ensure_collection()
        reranker = build_reranker(args.reranker_provider, args.reranker_model) if args.retrieval_mode == "hybrid_rerank" else None
        return None, HybridRetriever(embedding, None, store, reranker), embedding
    raise RuntimeError("No retrieval backend configured. Provide --chunks-path or --use-milvus, or pass --demo explicitly.")


def maybe_duckdb(args: Namespace) -> Any | None:
    if not getattr(args, "duckdb_path", None):
        return None
    if not Path(args.duckdb_path).exists():
        raise FileNotFoundError(f"DuckDB path does not exist: {args.duckdb_path}")
    return DuckDBStore(args.duckdb_path)


def run_pipeline(query: str, args: Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    duck = maybe_duckdb(args)
    _, retriever, embedding = build_runtime(args)
    if args.pipeline_mode == "agentic":
        state = AgenticRAG(duckdb_store=duck, retriever=retriever, embedding_provider=embedding, max_iterations=args.max_iterations, default_top_k=args.top_k, retrieval_mode=args.retrieval_mode, candidate_k=args.candidate_k).run(query)
        final_evidence = state.reranked_evidence or state.retrieved_evidence
        deterministic_answer = state.answer
        query_type = state.query_type or None
        expected_answer_type = state.expected_answer_type or None
        retrieval_route = state.retrieval_route
        evidence_gaps = state.evidence_gaps
        trace = state.trace
        tool_calls = state.tool_calls
        followups = sum(1 for step in trace if step == "followup_retrieval")
        retrieved, reranked = state.retrieved_evidence, state.reranked_evidence
    else:
        source_types = parse_source_types(args.source_types)
        retrieved = dedupe_chunks(retriever.retrieve(query, dense_vector=embedding.embed_query(query) if embedding else None, filters=None, top_k=args.top_k, source_types=source_types, mode=args.retrieval_mode, candidate_k=args.candidate_k))
        reranked = retrieved if args.retrieval_mode == "hybrid_rerank" else []
        final_evidence = reranked or retrieved
        plan = QueryPlan(query=query, query_type="general", expected_answer_type="general_answer", retrieval_route="hybrid", source_types=source_types, top_k=args.top_k)
        deterministic_answer, _ = synthesize_answer(plan, [], final_evidence, [])
        query_type = expected_answer_type = None
        retrieval_route = "direct_retrieval"
        evidence_gaps = []
        trace = ["basic_rag_direct_retrieval", "basic_rag_rerank_if_enabled", "final_answer"]
        tool_calls = [{"tool": "hybrid_retriever", "payload": {"pipeline_mode": "basic_rag", "retrieval_mode": args.retrieval_mode, "top_k": args.top_k, "candidate_k": args.candidate_k, "source_types": source_types}}]
        followups = 0
    llm_answer = llm_error = None
    if args.use_llm:
        llm_answer, llm_error = generate_grounded_answer(query, deterministic_answer, final_evidence, args.llm_provider, args.llm_model)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return {"query": query, "pipeline_mode": args.pipeline_mode, "agent_enabled": args.pipeline_mode == "agentic", "query_type": query_type, "expected_answer_type": expected_answer_type, "retrieval_route": retrieval_route, "retrieval_mode": args.retrieval_mode, "top_k": args.top_k, "candidate_k": args.candidate_k, "reranker_provider": args.reranker_provider, "deterministic_answer": deterministic_answer, "llm_provider": args.llm_provider if args.use_llm else None, "llm_model": args.llm_model if args.use_llm else None, "llm_answer": llm_answer, "llm_error": llm_error, "evidence_gaps": evidence_gaps, "retrieved_evidence": [compact_evidence(c) for c in retrieved], "reranked_evidence": [compact_evidence(c) for c in reranked], "evidence_manifest": evidence_manifest(final_evidence), "trace": trace, "tool_calls": tool_calls, "latency_ms": latency_ms, "followup_query_count": followups, "tool_call_count": len(tool_calls), "retrieved_evidence_count": len(retrieved), "reranked_evidence_count": len(reranked), "final_evidence_count": len(final_evidence)}


def print_result(result: dict[str, Any], output_json: bool, show_trace: bool, show_sources: bool = True) -> None:
    if output_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result.get("llm_answer") or result["deterministic_answer"])
    if show_sources:
        for item in result["evidence_manifest"]:
            print(f"- {item['source_type']}: {item['title']} ({item['chunk_id']})")
    if show_trace:
        print("Trace: " + " -> ".join(result["trace"]))
