#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

STAGES = [
    "dense_first_stage",
    "bm25_first_stage",
    "rrf_fused",
    "reranked",
    "final",
]

def dedupe_keep_order(items):
    seen = set()
    out = []
    for x in items or []:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out

def dcg_from_grades(grades):
    total = 0.0
    for i, g in enumerate(grades, start=1):
        total += (2 ** float(g) - 1.0) / math.log2(i + 1)
    return total

def load_eval_gold(path):
    gold = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)
            qid = x.get("id") or x.get("query_id")

            relevance_grades = {}
            for cid, g in (x.get("relevance_grades") or {}).items():
                try:
                    relevance_grades[cid] = float(g)
                except Exception:
                    relevance_grades[cid] = 1.0

            gold[qid] = {
                "query": x.get("query"),
                "relevant_chunk_ids": dedupe_keep_order(x.get("relevant_chunk_ids", [])),
                "relevant_doc_ids": dedupe_keep_order(x.get("relevant_doc_ids", [])),
                "relevance_grades": relevance_grades,
                "expected_source_types": x.get("expected_source_types", []),
                "expected_source_subtypes": x.get("expected_source_subtypes", []),
            }

    return gold

def collect_needed_chunk_ids(results_path):
    needed = set()

    with open(results_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            for ids in (x.get("stage_chunk_ids") or {}).values():
                needed.update(ids or [])

            for h in x.get("hits", []) or []:
                cid = h.get("chunk_id")
                if cid:
                    needed.add(cid)

    return needed

def load_chunk_meta(chunks_path, needed_ids):
    meta = {}

    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            cid = x.get("chunk_id")
            if cid in needed_ids:
                meta[cid] = {
                    "doc_id": x.get("doc_id"),
                    "source_type": x.get("source_type"),
                    "source_subtype": x.get("source_subtype"),
                    "title": x.get("title"),
                }

    return meta

def ids_to_doc_ids(chunk_ids, chunk_meta):
    doc_ids = []
    for cid in chunk_ids:
        did = chunk_meta.get(cid, {}).get("doc_id")
        if did:
            doc_ids.append(did)
    return dedupe_keep_order(doc_ids)

def true_metrics_at_k(chunk_ids, gold_item, chunk_meta, k):
    """
    Correct metrics:
    - chunk-level qrels优先
    - 如果没有 relevant_chunk_ids，则使用 relevant_doc_ids
    - 同一个 relevant chunk/doc 只计一次
    - Recall/nDCG 永远不会超过 1
    """

    retrieved_chunk_ids = dedupe_keep_order(chunk_ids)[:k]

    rel_chunks = set(gold_item.get("relevant_chunk_ids") or [])
    rel_docs = set(gold_item.get("relevant_doc_ids") or [])
    relevance_grades = gold_item.get("relevance_grades") or {}

    # -------- chunk-level gold --------
    if rel_chunks:
        retrieved_units = retrieved_chunk_ids
        relevant_units = rel_chunks

        unit_grade = {}
        for cid in rel_chunks:
            unit_grade[cid] = float(relevance_grades.get(cid, 1.0))

    # -------- doc-level gold --------
    elif rel_docs:
        retrieved_units = ids_to_doc_ids(retrieved_chunk_ids, chunk_meta)
        relevant_units = rel_docs

        unit_grade = {}
        for did in rel_docs:
            unit_grade[did] = 1.0

    # -------- no qrels --------
    else:
        return {
            f"Precision@{k}": None,
            f"Recall@{k}": None,
            f"HitRate@{k}": None,
            f"MRR@{k}": None,
            f"nDCG@{k}": None,
            f"RelevantFound@{k}": None,
            f"RelevantTotal": 0,
        }

    retrieved_units = dedupe_keep_order(retrieved_units)
    seen_relevant = set()

    binary_rels = []
    graded_rels = []

    first_relevant_rank = None

    for rank, unit in enumerate(retrieved_units, start=1):
        if unit in relevant_units and unit not in seen_relevant:
            seen_relevant.add(unit)
            binary_rels.append(1.0)
            graded_rels.append(float(unit_grade.get(unit, 1.0)))
            if first_relevant_rank is None:
                first_relevant_rank = rank
        else:
            binary_rels.append(0.0)
            graded_rels.append(0.0)

    relevant_found = len(seen_relevant)
    relevant_total = len(relevant_units)

    precision = relevant_found / k if k > 0 else 0.0
    recall = relevant_found / relevant_total if relevant_total > 0 else 0.0
    hit_rate = 1.0 if relevant_found > 0 else 0.0
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    dcg = dcg_from_grades(graded_rels)

    ideal_grades = sorted(
        [float(unit_grade.get(unit, 1.0)) for unit in relevant_units],
        reverse=True,
    )[:k]

    idcg = dcg_from_grades(ideal_grades)

    ndcg = dcg / idcg if idcg > 0 else 0.0

    # safety clamp
    recall = min(max(recall, 0.0), 1.0)
    ndcg = min(max(ndcg, 0.0), 1.0)

    return {
        f"Precision@{k}": precision,
        f"Recall@{k}": recall,
        f"HitRate@{k}": hit_rate,
        f"MRR@{k}": mrr,
        f"nDCG@{k}": ndcg,
        f"RelevantFound@{k}": relevant_found,
        f"RelevantTotal": relevant_total,
    }

def coverage_at_k(chunk_ids, gold_item, chunk_meta, k):
    retrieved = dedupe_keep_order(chunk_ids)[:k]

    expected_types = set(gold_item.get("expected_source_types") or [])
    expected_subtypes = set(gold_item.get("expected_source_subtypes") or [])

    got_types = {
        chunk_meta.get(cid, {}).get("source_type")
        for cid in retrieved
        if chunk_meta.get(cid, {}).get("source_type")
    }

    got_subtypes = {
        chunk_meta.get(cid, {}).get("source_subtype")
        for cid in retrieved
        if chunk_meta.get(cid, {}).get("source_subtype")
    }

    return {
        f"SourceTypeCoverage@{k}": len(got_types & expected_types) / len(expected_types) if expected_types else None,
        f"SourceSubtypeCoverage@{k}": len(got_subtypes & expected_subtypes) / len(expected_subtypes) if expected_subtypes else None,
    }

def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-path", required=True)
    ap.add_argument("--results-path", required=True)
    ap.add_argument("--chunks-path", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--per-query-csv", required=True)
    ap.add_argument("--ks", default="5,10,20")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    gold = load_eval_gold(args.eval_path)

    needed_ids = collect_needed_chunk_ids(args.results_path)
    chunk_meta = load_chunk_meta(args.chunks_path, needed_ids)

    print("gold queries:", len(gold))
    print("needed chunk ids:", len(needed_ids))
    print("chunk meta loaded:", len(chunk_meta))

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.per_query_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    agg = defaultdict(lambda: defaultdict(list))

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        with open(args.results_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)
                qid = row.get("id") or row.get("query_id")

                gold_item = gold.get(qid)
                if not gold_item:
                    continue

                stage_chunk_ids = row.get("stage_chunk_ids") or {}

                if "final" not in stage_chunk_ids:
                    stage_chunk_ids["final"] = [
                        h.get("chunk_id")
                        for h in row.get("hits", []) or []
                        if h.get("chunk_id")
                    ]

                true_stage_metrics = {}

                for stage in STAGES:
                    ids = stage_chunk_ids.get(stage, []) or []

                    stage_metrics = {
                        "count": len(dedupe_keep_order(ids)),
                    }

                    for k in ks:
                        stage_metrics.update(true_metrics_at_k(ids, gold_item, chunk_meta, k))
                        stage_metrics.update(coverage_at_k(ids, gold_item, chunk_meta, k))

                    true_stage_metrics[stage] = stage_metrics

                    for name, value in stage_metrics.items():
                        if isinstance(value, (int, float)):
                            agg[stage][name].append(value)

                row["true_stage_metrics"] = true_stage_metrics
                rows.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "eval_path": args.eval_path,
        "results_path": args.results_path,
        "chunks_path": args.chunks_path,
        "n": len(rows),
        "ks": ks,
        "stage_metric_means": {
            stage: {
                name: mean(values)
                for name, values in metrics.items()
            }
            for stage, metrics in agg.items()
        },
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "metric", "mean"])
        for stage, metrics in summary["stage_metric_means"].items():
            for metric, value in sorted(metrics.items()):
                w.writerow([stage, metric, value])

    fields = [
        "id",
        "mode",
        "query",
        "stage",
        "count",
        "RelevantTotal",
        "RelevantFound@5",
        "Precision@5",
        "Recall@5",
        "HitRate@5",
        "MRR@5",
        "nDCG@5",
        "SourceTypeCoverage@5",
        "SourceSubtypeCoverage@5",
        "RelevantFound@10",
        "Precision@10",
        "Recall@10",
        "HitRate@10",
        "MRR@10",
        "nDCG@10",
        "RelevantFound@20",
        "Precision@20",
        "Recall@20",
        "HitRate@20",
        "MRR@20",
        "nDCG@20",
    ]

    with open(args.per_query_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for row in rows:
            for stage, m in row.get("true_stage_metrics", {}).items():
                w.writerow({
                    "id": row.get("id"),
                    "mode": row.get("mode"),
                    "query": row.get("query"),
                    "stage": stage,
                    "count": m.get("count"),
                    "RelevantTotal": m.get("RelevantTotal"),
                    "RelevantFound@5": m.get("RelevantFound@5"),
                    "Precision@5": m.get("Precision@5"),
                    "Recall@5": m.get("Recall@5"),
                    "HitRate@5": m.get("HitRate@5"),
                    "MRR@5": m.get("MRR@5"),
                    "nDCG@5": m.get("nDCG@5"),
                    "SourceTypeCoverage@5": m.get("SourceTypeCoverage@5"),
                    "SourceSubtypeCoverage@5": m.get("SourceSubtypeCoverage@5"),
                    "RelevantFound@10": m.get("RelevantFound@10"),
                    "Precision@10": m.get("Precision@10"),
                    "Recall@10": m.get("Recall@10"),
                    "HitRate@10": m.get("HitRate@10"),
                    "MRR@10": m.get("MRR@10"),
                    "nDCG@10": m.get("nDCG@10"),
                    "RelevantFound@20": m.get("RelevantFound@20"),
                    "Precision@20": m.get("Precision@20"),
                    "Recall@20": m.get("Recall@20"),
                    "HitRate@20": m.get("HitRate@20"),
                    "MRR@20": m.get("MRR@20"),
                    "nDCG@20": m.get("nDCG@20"),
                })

    print("wrote", args.output_jsonl)
    print("wrote", args.summary_json)
    print("wrote", args.summary_csv)
    print("wrote", args.per_query_csv)

if __name__ == "__main__":
    main()
