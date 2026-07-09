#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MODES = ["dense_only", "bm25_only", "hybrid_rrf", "hybrid_rerank"]
STAGES = ["dense_first_stage", "bm25_first_stage", "rrf_fused", "reranked", "final"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []

    for mode in MODES:
        p = root / f"results_{mode}" / "true_metrics_summary.json"
        if not p.exists():
            print("missing", p)
            continue

        x = json.load(open(p, encoding="utf-8"))
        stage_means = x.get("stage_metric_means", {})

        for stage in STAGES:
            metrics = stage_means.get(stage)
            if not metrics:
                continue

            rows.append({
                "mode": mode,
                "stage": stage,
                "count_mean": metrics.get("count"),
                "RelevantTotal": metrics.get("RelevantTotal"),

                "RelevantFound@5": metrics.get("RelevantFound@5"),
                "Precision@5": metrics.get("Precision@5"),
                "Recall@5": metrics.get("Recall@5"),
                "HitRate@5": metrics.get("HitRate@5"),
                "MRR@5": metrics.get("MRR@5"),
                "nDCG@5": metrics.get("nDCG@5"),
                "SourceTypeCoverage@5": metrics.get("SourceTypeCoverage@5"),
                "SourceSubtypeCoverage@5": metrics.get("SourceSubtypeCoverage@5"),

                "RelevantFound@10": metrics.get("RelevantFound@10"),
                "Precision@10": metrics.get("Precision@10"),
                "Recall@10": metrics.get("Recall@10"),
                "HitRate@10": metrics.get("HitRate@10"),
                "MRR@10": metrics.get("MRR@10"),
                "nDCG@10": metrics.get("nDCG@10"),

                "RelevantFound@20": metrics.get("RelevantFound@20"),
                "Precision@20": metrics.get("Precision@20"),
                "Recall@20": metrics.get("Recall@20"),
                "HitRate@20": metrics.get("HitRate@20"),
                "MRR@20": metrics.get("MRR@20"),
                "nDCG@20": metrics.get("nDCG@20"),
            })

    if not rows:
        raise SystemExit("No rows found.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("wrote", args.output)

    for r in rows:
        if r["stage"] == "final":
            print(
                r["mode"],
                "Final@5:",
                "P=", r["Precision@5"],
                "R=", r["Recall@5"],
                "MRR=", r["MRR@5"],
                "nDCG=", r["nDCG@5"],
            )

if __name__ == "__main__":
    main()
