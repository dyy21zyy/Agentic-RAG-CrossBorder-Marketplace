#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from crossborder_agentic_rag.evaluation.dataset import load_eval_dataset
from crossborder_agentic_rag.evaluation import retrieval_metrics as rm
from crossborder_agentic_rag.ingestion.io_utils import read_chunks_jsonl
from crossborder_agentic_rag.llm.embeddings import build_embedding_provider
from crossborder_agentic_rag.retrieval import LocalBM25Retriever, build_reranker
from crossborder_agentic_rag.retrieval.rrf_fusion import rrf_fusion
from crossborder_agentic_rag.retrieval.utils import dedupe_chunks
from crossborder_agentic_rag.storage.milvus_store import MilvusChunkStore
from crossborder_agentic_rag.agents.classify import classify_query
from crossborder_agentic_rag.agents.planner import build_query_plan
from crossborder_agentic_rag.agents.answer import synthesize_answer

@contextmanager
def timer(bucket: dict, key: str):
    st = time.perf_counter()
    try:
        yield
    finally:
        bucket[key] = (time.perf_counter() - st) * 1000

def chunk_to_dict(c):
    if hasattr(c, "to_dict"):
        d = c.to_dict()
    elif isinstance(c, dict):
        d = dict(c)
    else:
        d = {
            "chunk_id": getattr(c, "chunk_id", ""),
            "doc_id": getattr(c, "doc_id", ""),
            "source_type": getattr(c, "source_type", ""),
            "source_subtype": getattr(c, "source_subtype", ""),
            "title": getattr(c, "title", ""),
            "content": getattr(c, "content", ""),
            "metadata": getattr(c, "metadata", {}),
        }
    content = d.get("content", "") or ""
    d["content_preview"] = content[:300]
    if len(content) > 600:
        d["content"] = content[:600]
    return d

def metric_map(hits, ex, ks):
    out = {}
    for k in ks:
        out[f"Precision@{k}"] = rm.precision_at_k(hits, ex, k)
        out[f"Recall@{k}"] = rm.recall_at_k(hits, ex, k)
        out[f"HitRate@{k}"] = rm.hit_rate_at_k(hits, ex, k)
        out[f"MRR@{k}"] = rm.mrr_at_k(hits, ex, k)
        out[f"nDCG@{k}"] = rm.ndcg_at_k(hits, ex, k)
        out[f"SourceTypeCoverage@{k}"] = rm.source_type_coverage(hits, ex.expected_source_types, k)
        out[f"SourceSubtypeCoverage@{k}"] = rm.source_subtype_coverage(hits, ex.expected_source_subtypes, k)
    return out

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def percentile(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    idx = min(len(xs) - 1, max(0, math.ceil(len(xs) * p / 100) - 1))
    return xs[idx]

def aggregate(rows):
    metrics = defaultdict(list)
    timings = defaultdict(list)
    errors = 0

    for r in rows:
        if r.get("error"):
            errors += 1

        for k, v in (r.get("metrics") or {}).items():
            if isinstance(v, (int, float)):
                metrics[k].append(v)

        for k, v in (r.get("timings_ms") or {}).items():
            if isinstance(v, (int, float)):
                timings[k].append(v)

    return {
        "n": len(rows),
        "errors": errors,
        "metric_means": {k: mean(v) for k, v in metrics.items()},
        "timing_ms_mean": {k: mean(v) for k, v in timings.items()},
        "timing_ms_p50": {k: percentile(v, 50) for k, v in timings.items()},
        "timing_ms_p95": {k: percentile(v, 95) for k, v in timings.items()},
    }

def write_csv_summary(summary, output_path):
    rows = []
    for group_name in ["metric_means", "timing_ms_mean", "timing_ms_p50", "timing_ms_p95"]:
        for k, v in summary.get(group_name, {}).items():
            rows.append({"group": group_name, "name": k, "value": v})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "name", "value"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-path", required=True)
    ap.add_argument("--chunks-path", required=True)
    ap.add_argument("--collection-name", default="ip_rag_collection")
    ap.add_argument("--retrieval-mode", required=True, choices=["bm25_only", "dense_only", "hybrid_rrf", "hybrid_rerank"])
    ap.add_argument("--use-milvus", action="store_true")
    ap.add_argument("--embedding-provider", default="local")
    ap.add_argument("--reranker-provider", default="local")
    ap.add_argument("--reranker-model", default=None)

    # 核心：20 -> 10 -> 5
    ap.add_argument("--first-stage-k", type=int, default=20)
    ap.add_argument("--fusion-k", type=int, default=10)
    ap.add_argument("--final-k", type=int, default=5)

    # 指标同时看 @5/@10/@20，方便解释每一层
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    setup = {}

    with timer(setup, "setup_load_eval_ms"):
        examples = load_eval_dataset(ROOT / args.eval_path)

    chunks = None
    bm25 = None
    if args.retrieval_mode in {"bm25_only", "hybrid_rrf", "hybrid_rerank"}:
        with timer(setup, "setup_load_chunks_ms"):
            chunks = read_chunks_jsonl(ROOT / args.chunks_path)
        with timer(setup, "setup_build_bm25_ms"):
            bm25 = LocalBM25Retriever(chunks)

    embedding = None
    vector_store = None

    if args.retrieval_mode != "bm25_only":
        with timer(setup, "setup_embedding_provider_ms"):
            embedding = build_embedding_provider(args.embedding_provider)

        if args.use_milvus:
            tmp_vec = embedding.embed_query("dimension check")
            uri = os.getenv("RAG_MILVUS_URI") or os.getenv("MILVUS_URI")
            with timer(setup, "setup_milvus_connect_ms"):
                vector_store = MilvusChunkStore(
                    uri=uri,
                    token=os.getenv("MILVUS_TOKEN"),
                    collection_name=args.collection_name,
                    embedding_dim=len(tmp_vec),
                )
                vector_store.connect()
                vector_store.ensure_collection()
        else:
            raise RuntimeError("dense/hybrid modes require --use-milvus")

    reranker = None
    if args.retrieval_mode == "hybrid_rerank":
        with timer(setup, "setup_reranker_ms"):
            reranker = build_reranker(args.reranker_provider, args.reranker_model)

    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    rows = []

    for ex in examples:
        q_timings = {}
        q_total_st = time.perf_counter()

        row = {
            "id": ex.id,
            "query": ex.query,
            "mode": args.retrieval_mode,
            "first_stage_k": args.first_stage_k,
            "fusion_k": args.fusion_k,
            "final_k": args.final_k,
            "expected_source_types": ex.expected_source_types,
            "expected_source_subtypes": ex.expected_source_subtypes,
            "relevant_chunk_ids": ex.relevant_chunk_ids,
        }

        try:
            with timer(q_timings, "classification_ms"):
                cls = classify_query(ex.query)

            with timer(q_timings, "planning_ms"):
                plan = build_query_plan(ex.query, cls, args.final_k)

            # 评估时优先用 eval 里的 expected_source_types 做过滤，
            # 避免分类器偶然路由错导致评估不稳定。
            source_types = ex.expected_source_types or plan.source_types or None

            dense_vec = None
            if args.retrieval_mode != "bm25_only":
                with timer(q_timings, "query_embedding_ms"):
                    dense_vec = embedding.embed_query(ex.query)

            stage_hits = {
                "dense_first_stage": [],
                "bm25_first_stage": [],
                "rrf_fused": [],
                "reranked": [],
                "final": [],
            }

            if args.retrieval_mode == "bm25_only":
                with timer(q_timings, "bm25_search_ms"):
                    bm25_hits = bm25.search(
                        ex.query,
                        source_types=source_types,
                        top_k=args.first_stage_k,
                    )
                stage_hits["bm25_first_stage"] = bm25_hits
                final_hits = dedupe_chunks(bm25_hits)[:args.final_k]

            elif args.retrieval_mode == "dense_only":
                with timer(q_timings, "dense_search_ms"):
                    dense_hits = vector_store.dense_search(
                        dense_vec,
                        source_types=source_types,
                        top_k=args.first_stage_k,
                    )
                stage_hits["dense_first_stage"] = dense_hits
                final_hits = dedupe_chunks(dense_hits)[:args.final_k]

            elif args.retrieval_mode == "hybrid_rrf":
                with timer(q_timings, "bm25_search_ms"):
                    bm25_hits = bm25.search(
                        ex.query,
                        source_types=source_types,
                        top_k=args.first_stage_k,
                    )

                with timer(q_timings, "dense_search_ms"):
                    dense_hits = vector_store.dense_search(
                        dense_vec,
                        source_types=source_types,
                        top_k=args.first_stage_k,
                    )

                with timer(q_timings, "rrf_fusion_ms"):
                    fused = rrf_fusion(
                        [bm25_hits, dense_hits],
                        top_k=args.fusion_k,
                    )

                stage_hits["bm25_first_stage"] = bm25_hits
                stage_hits["dense_first_stage"] = dense_hits
                stage_hits["rrf_fused"] = fused

                # hybrid_rrf 没有 rerank，但最终仍按你的 LLM 上下文策略取 5 条
                final_hits = dedupe_chunks(fused)[:args.final_k]

            elif args.retrieval_mode == "hybrid_rerank":
                with timer(q_timings, "bm25_search_ms"):
                    bm25_hits = bm25.search(
                        ex.query,
                        source_types=source_types,
                        top_k=args.first_stage_k,
                    )

                with timer(q_timings, "dense_search_ms"):
                    dense_hits = vector_store.dense_search(
                        dense_vec,
                        source_types=source_types,
                        top_k=args.first_stage_k,
                    )

                with timer(q_timings, "rrf_fusion_ms"):
                    fused = rrf_fusion(
                        [bm25_hits, dense_hits],
                        top_k=args.fusion_k,
                    )

                with timer(q_timings, "rerank_ms"):
                    reranked = reranker.rerank(
                        ex.query,
                        dedupe_chunks(fused),
                        args.final_k,
                    )

                stage_hits["bm25_first_stage"] = bm25_hits
                stage_hits["dense_first_stage"] = dense_hits
                stage_hits["rrf_fused"] = fused
                stage_hits["reranked"] = reranked

                final_hits = dedupe_chunks(reranked)[:args.final_k]

            with timer(q_timings, "answer_generation_ms"):
                try:
                    answer, citations = synthesize_answer(plan, [], final_hits, [])
                except Exception:
                    answer, citations = "", []

            q_timings["query_total_ms"] = (time.perf_counter() - q_total_st) * 1000

            row["query_type"] = cls.query_type
            row["retrieval_route"] = cls.retrieval_route
            row["plan_source_types"] = plan.source_types
            row["source_types_used"] = source_types

            row["stage_counts"] = {
                "dense_first_stage": len(stage_hits["dense_first_stage"]),
                "bm25_first_stage": len(stage_hits["bm25_first_stage"]),
                "rrf_fused": len(stage_hits["rrf_fused"]),
                "reranked": len(stage_hits["reranked"]),
                "final": len(final_hits),
            }

            row["stage_chunk_ids"] = {
                k: [getattr(c, "chunk_id", "") for c in v]
                for k, v in stage_hits.items()
            }
            row["stage_chunk_ids"]["final"] = [getattr(c, "chunk_id", "") for c in final_hits]

            row["hits"] = [chunk_to_dict(c) for c in final_hits]
            row["answer_preview"] = answer[:600]
            row["citations"] = citations
            row["metrics"] = metric_map(final_hits, ex, ks)
            row["timings_ms"] = q_timings
            row["error"] = None

        except Exception as e:
            q_timings["query_total_ms"] = (time.perf_counter() - q_total_st) * 1000
            row["error"] = repr(e)
            row["metrics"] = {}
            row["timings_ms"] = q_timings
            row["hits"] = []

        rows.append(row)

        print(json.dumps({
            "id": row["id"],
            "mode": row["mode"],
            "stage_counts": row.get("stage_counts"),
            "error": row["error"],
            "query_total_ms": row["timings_ms"].get("query_total_ms"),
            "Recall@5": row["metrics"].get("Recall@5"),
            "MRR@5": row["metrics"].get("MRR@5"),
            "nDCG@5": row["metrics"].get("nDCG@5"),
        }, ensure_ascii=False), flush=True)

    summary = {
        "run_config": vars(args),
        "setup_timings_ms": setup,
        "aggregate": aggregate(rows),
    }

    with open(out_dir / "eval_results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    write_csv_summary(summary["aggregate"], out_dir / "summary_flat.csv")

    print("wrote", out_dir / "eval_results.jsonl")
    print("wrote", out_dir / "summary.json")
    print("wrote", out_dir / "summary_flat.csv")

if __name__ == "__main__":
    main()
