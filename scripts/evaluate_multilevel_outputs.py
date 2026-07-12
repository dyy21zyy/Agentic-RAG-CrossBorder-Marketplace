#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path
from statistics import mean

CASE_PAT = re.compile(r"case\s+([0-9]+:\d{2}-cv-\d{5})", re.I)
PATENT_PAT = re.compile(r"patent\s+(\d{7,12})", re.I)

def safe_float(x, default=0.0):
    return x if isinstance(x, (int, float)) else default

def case_number_from_query(q: str):
    m = CASE_PAT.search(q or "")
    return m.group(1) if m else None

def patent_id_from_query(q: str):
    m = PATENT_PAT.search(q or "")
    return m.group(1) if m else None

def duckdb_structured_hit(row):
    tools = str(row.get("tools", ""))
    q = str(row.get("query", ""))
    ans = str(row.get("answer_preview", ""))
    case_number = case_number_from_query(q)

    if not case_number:
        return 0

    if "duckdb_lookup_tool" not in tools:
        return 0

    # 当前 comparison_outputs 没有完整 tool_calls，只能先基于 answer_preview 做后处理判断
    if case_number in ans and "Case / Litigation Evidence" in ans:
        return 1

    return 0

def patent_entity_hit(row):
    q = str(row.get("query", ""))
    ans = str(row.get("answer_preview", ""))
    patent_id = patent_id_from_query(q)

    if not patent_id:
        return 0

    # 判断答案里是否出现目标 patent id
    return 1 if patent_id in ans else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.outputs, encoding="utf-8")]

    strict_hit = []
    strict_recall = []
    duckdb_hits = []
    patent_entity_hits = []
    multilevel_hits = []
    multilevel_recalls = []

    by_type = {}

    for r in rows:
        q = str(r.get("query", ""))
        tools = str(r.get("tools", ""))

        hit = safe_float(r.get("hit_at_5"))
        recall = safe_float(r.get("recall_at_5"))

        d_hit = duckdb_structured_hit(r)
        p_hit = patent_entity_hit(r)

        strict_hit.append(hit)
        strict_recall.append(recall)
        duckdb_hits.append(d_hit)
        patent_entity_hits.append(p_hit)

        is_litigation_case = case_number_from_query(q) is not None and "litigation_search_tool" in tools
        is_patent_case = patent_id_from_query(q) is not None and "patent_search_tool" in tools

        # multi-level:
        # 1. 普通样本沿用 chunk-level hit/recall
        # 2. litigation case query 如果 DuckDB 命中，则认为 structured-level 命中
        # 3. patent query 暂时只额外统计 entity_hit，不直接替代 recall
        if is_litigation_case:
            ml_hit = max(hit, d_hit)
            ml_recall = max(recall, d_hit)
            group = "litigation_case"
        elif is_patent_case:
            ml_hit = hit
            ml_recall = recall
            group = "patent"
        elif "graph_rag_tool" in tools:
            ml_hit = hit
            ml_recall = recall
            group = "graph"
        elif "trademark_search_tool" in tools:
            ml_hit = hit
            ml_recall = recall
            group = "trademark"
        elif "litigation_search_tool" in tools:
            ml_hit = hit
            ml_recall = recall
            group = "litigation_other"
        else:
            ml_hit = hit
            ml_recall = recall
            group = "other"

        multilevel_hits.append(ml_hit)
        multilevel_recalls.append(ml_recall)

        by_type.setdefault(group, []).append({
            "strict_hit": hit,
            "strict_recall": recall,
            "duckdb_structured_hit": d_hit,
            "patent_entity_hit": p_hit,
            "multilevel_hit": ml_hit,
            "multilevel_recall": ml_recall,
            "latency_ms": safe_float(r.get("latency_ms")),
            "retrieval_ms": safe_float(r.get("retrieval_ms")),
        })

    def avg(xs, key=None):
        if key:
            vals = [x[key] for x in xs]
        else:
            vals = xs
        return round(mean(vals), 4) if vals else None

    summary = {
        "n": len(rows),
        "strict_hit_at_5_mean": avg(strict_hit),
        "strict_recall_at_5_mean": avg(strict_recall),
        "duckdb_structured_hit_mean_all": avg(duckdb_hits),
        "patent_entity_hit_mean_all": avg(patent_entity_hits),
        "multilevel_hit_at_5_mean": avg(multilevel_hits),
        "multilevel_recall_at_5_mean": avg(multilevel_recalls),
        "by_type": {}
    }

    for g, items in by_type.items():
        summary["by_type"][g] = {
            "n": len(items),
            "strict_hit_at_5_mean": avg(items, "strict_hit"),
            "strict_recall_at_5_mean": avg(items, "strict_recall"),
            "duckdb_structured_hit_mean": avg(items, "duckdb_structured_hit"),
            "patent_entity_hit_mean": avg(items, "patent_entity_hit"),
            "multilevel_hit_at_5_mean": avg(items, "multilevel_hit"),
            "multilevel_recall_at_5_mean": avg(items, "multilevel_recall"),
            "latency_ms_mean": avg(items, "latency_ms"),
            "retrieval_ms_mean": avg(items, "retrieval_ms"),
        }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
