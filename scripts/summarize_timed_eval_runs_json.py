#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.load(open(p, encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.root)

    modes = {
        "dense_only": "results_dense_only",
        "bm25_only": "results_bm25_only",
        "hybrid_rrf": "results_hybrid_rrf",
        "hybrid_rerank": "results_hybrid_rerank",
    }

    rows = []

    for mode, dirname in modes.items():
        summary = load_json(root / dirname / "summary.json")
        time_info = load_json(root / "logs" / f"time_{mode}.json")

        if not summary:
            continue

        agg = summary.get("aggregate", {})
        metric = agg.get("metric_means", {})
        timing_mean = agg.get("timing_ms_mean", {})
        timing_p95 = agg.get("timing_ms_p95", {})
        setup = summary.get("setup_timings_ms", {})
        config = summary.get("run_config", {})

        row = {
            "mode": mode,
            "n": agg.get("n"),
            "errors": agg.get("errors"),

            "first_stage_k": config.get("first_stage_k"),
            "fusion_k": config.get("fusion_k"),
            "final_k": config.get("final_k"),

            "Precision@5": metric.get("Precision@5"),
            "Recall@5": metric.get("Recall@5"),
            "HitRate@5": metric.get("HitRate@5"),
            "MRR@5": metric.get("MRR@5"),
            "nDCG@5": metric.get("nDCG@5"),

            "Precision@10": metric.get("Precision@10"),
            "Recall@10": metric.get("Recall@10"),
            "HitRate@10": metric.get("HitRate@10"),
            "MRR@10": metric.get("MRR@10"),
            "nDCG@10": metric.get("nDCG@10"),

            "Precision@20": metric.get("Precision@20"),
            "Recall@20": metric.get("Recall@20"),
            "HitRate@20": metric.get("HitRate@20"),
            "MRR@20": metric.get("MRR@20"),
            "nDCG@20": metric.get("nDCG@20"),

            "SourceTypeCoverage@5": metric.get("SourceTypeCoverage@5"),
            "SourceSubtypeCoverage@5": metric.get("SourceSubtypeCoverage@5"),

            "setup_load_chunks_ms": setup.get("setup_load_chunks_ms"),
            "setup_build_bm25_ms": setup.get("setup_build_bm25_ms"),
            "setup_embedding_provider_ms": setup.get("setup_embedding_provider_ms"),
            "setup_milvus_connect_ms": setup.get("setup_milvus_connect_ms"),
            "setup_reranker_ms": setup.get("setup_reranker_ms"),

            "query_total_ms_mean": timing_mean.get("query_total_ms"),
            "query_total_ms_p95": timing_p95.get("query_total_ms"),
            "classification_ms_mean": timing_mean.get("classification_ms"),
            "planning_ms_mean": timing_mean.get("planning_ms"),
            "query_embedding_ms_mean": timing_mean.get("query_embedding_ms"),
            "bm25_search_ms_mean": timing_mean.get("bm25_search_ms"),
            "dense_search_ms_mean": timing_mean.get("dense_search_ms"),
            "rrf_fusion_ms_mean": timing_mean.get("rrf_fusion_ms"),
            "rerank_ms_mean": timing_mean.get("rerank_ms"),
            "answer_generation_ms_mean": timing_mean.get("answer_generation_ms"),

            "wall_seconds": time_info.get("wall_seconds") if time_info else None,
            "wall_minutes": time_info.get("wall_minutes") if time_info else None,
            "max_rss_kb": time_info.get("max_rss_kb") if time_info else None,
        }

        rows.append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    keys = list(rows[0].keys()) if rows else []
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("wrote", args.output)
    for r in rows:
        print(r)

if __name__ == "__main__":
    main()
