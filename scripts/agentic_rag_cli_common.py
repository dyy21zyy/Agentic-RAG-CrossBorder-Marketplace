"""Shared helpers for basic RAG, rule-based Agentic RAG, and LLM-driven Agentic RAG CLIs."""
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
from crossborder_agentic_rag.agents.llm_agentic_rag import LLMAgenticRAG
from crossborder_agentic_rag.agents.llm_answer import (
    build_evidence_context,
    generate_grounded_answer as llm_generate_grounded_answer,
)
from crossborder_agentic_rag.ingestion.io_utils import read_chunks_jsonl
from crossborder_agentic_rag.llm.chat_client import build_chat_client
from crossborder_agentic_rag.llm.embeddings import FakeEmbeddingProvider, build_embedding_provider
from crossborder_agentic_rag.retrieval import HybridRetriever, LocalBM25Retriever, build_reranker
from crossborder_agentic_rag.retrieval.utils import dedupe_chunks
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
from crossborder_agentic_rag.schemas.queries import QueryPlan
from crossborder_agentic_rag.storage.duckdb_store import DuckDBStore
from crossborder_agentic_rag.storage.milvus_store import MilvusChunkStore
from crossborder_agentic_rag.tools.ip_tools import build_ip_tools
from crossborder_agentic_rag.graph import GraphRetriever


DEFAULT_SOURCE_TYPES = ["trademark", "patent", "litigation"]
DENSE_MODES = {"dense_only", "hybrid_rrf", "hybrid_rerank"}
HYBRID_MODES = {"hybrid_rrf", "hybrid_rerank"}


def load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def demo_chunks() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            "tm:mercedes:0",
            "tm-mercedes",
            "trademark",
            "classes",
            "MERCEDES trademark",
            "MERCEDES registration covers Nice class 12 vehicles and related goods.",
            {"chunk_index": 0},
        ),
        EvidenceChunk(
            "pat:bag:0",
            "pat-smart-bag",
            "patent",
            "claims",
            "Smart bag patent",
            "A smart travel bag patent claim describes location tracking and locking controls for luggage.",
            {"chunk_index": 0},
        ),
        EvidenceChunk(
            "lit:case:0",
            "lit-case",
            "litigation",
            "docket",
            "Marketplace IP case",
            "A marketplace case discusses seller infringement allegations and platform takedown evidence.",
            {"chunk_index": 0},
        ),
    ]


def parse_source_types(value: str | None) -> list[str]:
    return [x.strip() for x in (value or ",".join(DEFAULT_SOURCE_TYPES)).split(",") if x.strip()]


def compact_evidence(chunk: EvidenceChunk) -> dict[str, Any]:
    data = chunk.to_dict() if hasattr(chunk, "to_dict") else dict(chunk)
    return {
        "chunk_id": data.get("chunk_id"),
        "doc_id": data.get("doc_id"),
        "source_type": data.get("source_type"),
        "source_subtype": data.get("source_subtype"),
        "title": data.get("title"),
        "score": data.get("score"),
        "metadata": data.get("metadata", {}),
        "content_preview": (data.get("content") or "")[:240],
    }


def evidence_manifest(evidence: list[EvidenceChunk]) -> list[dict[str, Any]]:
    _, manifest = build_evidence_context(evidence)
    return manifest


def generate_grounded_answer(
    query: str,
    deterministic_answer: str,
    evidence: list[EvidenceChunk],
    provider: str | None,
    model: str | None,
) -> tuple[str | None, str | None]:
    client = build_chat_client(provider=provider or "template", model=model)
    out = llm_generate_grounded_answer(query, evidence, client, pipeline_mode=None)
    return out.get("llm_answer") or None, out.get("llm_error")


_ORIGINAL_GENERATE_GROUNDED_ANSWER = generate_grounded_answer


class DemoVectorStore:
    def __init__(self, chunks: list[EvidenceChunk]) -> None:
        self.chunks = chunks

    def dense_search(self, dense_vector, filters=None, top_k: int = 20):
        return self.chunks[:top_k]


def maybe_duckdb(args: Namespace) -> Any | None:
    if not getattr(args, "duckdb_path", None):
        return None
    if not Path(args.duckdb_path).exists():
        raise FileNotFoundError(f"DuckDB path does not exist: {args.duckdb_path}")
    return DuckDBStore(args.duckdb_path)


def _milvus_uri() -> str | None:
    return os.getenv("RAG_MILVUS_URI") or os.getenv("MILVUS_URI")


def _is_agentic_mode(mode: str) -> bool:
    return mode in {"agentic", "rule_based", "agentic_llm"}


class RAGRuntime:
    def __init__(
        self,
        duckdb_store,
        retriever,
        embedding_provider,
        chat_client,
        args,
        agent=None,
    ):
        self.duckdb_store = duckdb_store
        self.retriever = retriever
        self.embedding_provider = embedding_provider
        self.chat_client = chat_client
        self.args = args
        self.agent = agent

    def run_query(self, query: str) -> dict[str, Any]:
        start = time.perf_counter()
        args = self.args

        if args.pipeline_mode in {"agentic", "rule_based", "agentic_llm"}:
            state = self.agent.run(query)
            final_evidence = state.reranked_evidence or state.retrieved_evidence

            deterministic_answer = state.answer
            query_type = state.query_type or None
            expected_answer_type = state.expected_answer_type or None
            retrieval_route = state.retrieval_route
            evidence_gaps = state.evidence_gaps
            trace = state.trace
            tool_calls = state.tool_calls

            followups = sum(
                1
                for step in trace
                if step in {"followup_retrieval", "followup_tool_retrieval"}
            )
            retrieved = state.retrieved_evidence
            reranked = state.reranked_evidence

        else:
            source_types = parse_source_types(args.source_types)
            dense = self.embedding_provider.embed_query(query) if self.embedding_provider else None

            retrieved = dedupe_chunks(
                self.retriever.retrieve(
                    query,
                    dense_vector=dense,
                    filters=None,
                    top_k=args.top_k,
                    source_types=source_types,
                    mode=args.retrieval_mode,
                    candidate_k=args.candidate_k,
            dense_k=getattr(args, "dense_k", 20),
            bm25_k=getattr(args, "bm25_k", 20),
            rrf_k=getattr(args, "rrf_k", 10),
                )
            )
            reranked = retrieved if args.retrieval_mode == "hybrid_rerank" else []
            final_evidence = reranked or retrieved

            plan = QueryPlan(
                query=query,
                query_type="general",
                expected_answer_type="general_answer",
                retrieval_route="hybrid",
                source_types=source_types,
                top_k=args.top_k,
            )
            deterministic_answer, _ = synthesize_answer(plan, [], final_evidence, [])

            query_type = expected_answer_type = None
            retrieval_route = "direct_retrieval"
            evidence_gaps = []
            trace = ["basic_rag_direct_retrieval", "basic_rag_rerank_if_enabled", "final_answer"]
            tool_calls = [
                {
                    "tool": "hybrid_retriever",
                    "payload": {
                        "pipeline_mode": "basic_rag",
                        "retrieval_mode": args.retrieval_mode,
                        "top_k": args.top_k,
                        "candidate_k": args.candidate_k,
                        "source_types": source_types,
                    },
                }
            ]
            followups = 0

        llm_answer = llm_error = None
        llm_provider = llm_model = None
        manifest = evidence_manifest(final_evidence)

        if getattr(args, "use_llm", False):
            if generate_grounded_answer is _ORIGINAL_GENERATE_GROUNDED_ANSWER and self.chat_client is not None:
                llm = llm_generate_grounded_answer(
                    query,
                    final_evidence,
                    self.chat_client,
                    query_type=query_type,
                    retrieval_route=retrieval_route,
                    evidence_gaps=evidence_gaps,
                    pipeline_mode=args.pipeline_mode,
                    max_evidence=getattr(args, "max_evidence_for_llm", 6),
                    max_chars_each=getattr(args, "max_chars_per_evidence", 450),
                    max_tokens=getattr(args, "llm_max_tokens", 800),
                    temperature=getattr(args, "temperature", 0.0),
                )
                llm_answer = llm["llm_answer"]
                llm_error = llm["llm_error"]
                manifest = llm["evidence_manifest"]
                llm_provider = llm["provider"]
                llm_model = llm["model"]
            else:
                llm_answer, llm_error = generate_grounded_answer(
                    query,
                    deterministic_answer,
                    final_evidence,
                    getattr(args, "llm_provider", None),
                    getattr(args, "llm_model", None),
                )
                llm_provider = getattr(args, "llm_provider", None)
                llm_model = getattr(args, "llm_model", None)
                _, manifest = build_evidence_context(
                    final_evidence,
                    getattr(args, "max_evidence_for_llm", 6),
                    getattr(args, "max_chars_per_evidence", 450),
                )

        return {
            "query": query,
            "pipeline_mode": args.pipeline_mode,
            "agent_enabled": _is_agentic_mode(args.pipeline_mode),
            "query_type": query_type,
            "expected_answer_type": expected_answer_type,
            "retrieval_route": retrieval_route,
            "retrieval_mode": args.retrieval_mode,
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "reranker_provider": args.reranker_provider,
            "deterministic_answer": deterministic_answer,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "llm_answer": llm_answer,
            "llm_error": llm_error,
            "evidence_gaps": evidence_gaps,
            "retrieved_evidence": [compact_evidence(c) for c in retrieved],
            "reranked_evidence": [compact_evidence(c) for c in reranked],
            "evidence_manifest": manifest,
            "trace": trace,
            "tool_calls": tool_calls,
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "followup_query_count": followups,
            "tool_call_count": len(tool_calls),
            "retrieved_evidence_count": len(retrieved),
            "reranked_evidence_count": len(reranked),
            "final_evidence_count": len(final_evidence),
        }




def build_graph_retriever_auto() -> Any | None:
    """Load NetworkX GraphRAG retriever from RAG_GRAPH_PATH if available."""
    graph_path = os.getenv("RAG_GRAPH_PATH") or os.getenv("GRAPH_PATH")

    candidates = []
    if graph_path:
        candidates.append(Path(graph_path))

    candidates.extend([
        Path("data/processed/graph_index_full/ip_graph.pkl"),
        Path("data/processed/graph_index/ip_graph.pkl"),
        Path("data/processed/ip_graph.pkl"),
    ])

    for p in candidates:
        if p.exists():
            try:
                print(f"[build_runtime] loading GraphRAG index: {p}", file=sys.stderr)
                return GraphRetriever.load(p)
            except Exception as exc:
                print(f"[build_runtime] failed to load GraphRAG index {p}: {exc}", file=sys.stderr)

    print("[build_runtime] GraphRAG index not found; graph_retriever=None", file=sys.stderr)
    return None


def build_runtime(args: Namespace) -> RAGRuntime:
    mode = args.retrieval_mode
    chunks: list[EvidenceChunk] = []

    if getattr(args, "demo", False):
        chunks = demo_chunks()
    elif getattr(args, "chunks_path", None):
        path = Path(args.chunks_path)
        if not path.is_file():
            raise FileNotFoundError(f"Chunks path does not exist: {path}")
        chunks = read_chunks_jsonl(path)

    if mode in {"bm25_only", *HYBRID_MODES} and not chunks:
        raise RuntimeError(
            "BM25 chunks are required for bm25_only and hybrid retrieval. "
            "Provide --chunks-path or --demo."
        )

    vector_store = None
    embedding = None

    if mode in DENSE_MODES:
        embedding = FakeEmbeddingProvider() if getattr(args, "demo", False) else build_embedding_provider(args.embedding_provider)

        if getattr(args, "demo", False):
            vector_store = DemoVectorStore(chunks)
        else:
            if not getattr(args, "use_milvus", False):
                raise RuntimeError(
                    "Dense vector store is required for dense/hybrid retrieval. "
                    "Pass --use-milvus and configure RAG_MILVUS_URI or MILVUS_URI."
                )
            uri = _milvus_uri()
            if not uri:
                raise RuntimeError(
                    "Milvus URI missing: set RAG_MILVUS_URI or MILVUS_URI for dense/hybrid retrieval."
                )
            dim = len(embedding.embed_query("dimension probe"))
            vector_store = MilvusChunkStore(
                uri,
                os.getenv("MILVUS_TOKEN"),
                args.collection_name,
                dim,
            )
            vector_store.connect()
            vector_store.ensure_collection()

    bm25 = LocalBM25Retriever(chunks) if chunks else None
    reranker = build_reranker(args.reranker_provider, args.reranker_model) if mode == "hybrid_rerank" else None
    retriever = HybridRetriever(embedding, bm25, vector_store, reranker)
    duck = maybe_duckdb(args)

    chat = (
        build_chat_client(
            provider=getattr(args, "llm_provider", None),
            base_url=getattr(args, "llm_base_url", None),
            model=getattr(args, "llm_model", None),
            default_temperature=getattr(args, "temperature", 0.0),
            default_max_tokens=getattr(args, "llm_max_tokens", 800),
        )
        if getattr(args, "use_llm", False)
        else None
    )

    agent = None

    if args.pipeline_mode in {"agentic", "rule_based"}:
        agent = AgenticRAG(
            duck,
            retriever,
            embedding,
            args.max_iterations,
            args.top_k,
            mode,
            args.candidate_k,
        )

    elif args.pipeline_mode == "agentic_llm":
        graph_retriever = build_graph_retriever_auto()
        tools = build_ip_tools(
            retriever=retriever,
            embedding_provider=embedding,
            duckdb_store=duck,
            graph_retriever=graph_retriever,
            default_top_k=args.top_k,
            candidate_k=args.candidate_k,
            dense_k=getattr(args, "dense_k", 20),
            bm25_k=getattr(args, "bm25_k", 20),
            rrf_k=getattr(args, "rrf_k", 10),
        )
        agent = LLMAgenticRAG(
            duckdb_store=duck,
            retriever=retriever,
            embedding_provider=embedding,
            graph_retriever=graph_retriever,
            llm=chat,
            tools=tools,
            max_iterations=args.max_iterations,
            default_top_k=args.top_k,
            retrieval_mode=mode,
            candidate_k=args.candidate_k,
            dense_k=getattr(args, "dense_k", 20),
            bm25_k=getattr(args, "bm25_k", 20),
            rrf_k=getattr(args, "rrf_k", 10),
        )
    return RAGRuntime(duck, retriever, embedding, chat, args, agent)


def run_pipeline(query: str, args: Namespace) -> dict[str, Any]:
    return build_runtime(args).run_query(query)


def print_result(
    result: dict[str, Any],
    output_json: bool,
    show_trace: bool,
    show_sources: bool = True,
) -> None:
    if output_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(result.get("llm_answer") or result["deterministic_answer"])

    if result.get("llm_error"):
        print(f"LLM error: {result['llm_error']}")

    if show_sources:
        for item in result.get("evidence_manifest", []):
            print(
                f"- [{item.get('evidence_id','')}] "
                f"{item.get('source_type')}/{item.get('source_subtype')}: "
                f"{item.get('title')} ({item.get('chunk_id')}) "
                f"score={item.get('score')} {str(item.get('content',''))[:80]}"
            )

    if show_trace:
        print("Trace: " + " -> ".join(result.get("trace", [])))
        if result.get("tool_calls"):
            print("Tool calls: " + json.dumps(result["tool_calls"], ensure_ascii=False))


# ===== FORMAL_RETRIEVAL_K_PATCH =====
def attach_formal_retrieval_k(obj, args):
    """Attach formal retrieval K settings to runtime, agent, and retriever objects."""
    if obj is None or args is None:
        return obj

    for target in [
        obj,
        getattr(obj, "agent", None),
        getattr(obj, "retriever", None),
        getattr(getattr(obj, "agent", None), "retriever", None),
    ]:
        if target is None:
            continue
        for name, default in [
            ("dense_k", 20),
            ("bm25_k", 20),
            ("rrf_k", 10),
            ("top_k", 5),
            ("candidate_k", 20),
        ]:
            try:
                setattr(target, name, getattr(args, name, default))
            except Exception:
                pass
    return obj
# ===== END FORMAL_RETRIEVAL_K_PATCH =====
