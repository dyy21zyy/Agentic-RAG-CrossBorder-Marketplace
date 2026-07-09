#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", required=True)
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-grade", type=int, default=2)
    args = ap.parse_args()

    templates = {}
    for line in open(args.templates, encoding="utf-8"):
        x = json.loads(line)
        qid = x.get("id") or x.get("query_id")
        x["id"] = qid
        templates[qid] = x

    rel = defaultdict(list)
    grades = defaultdict(dict)
    key_points = defaultdict(list)

    for line in open(args.judgments, encoding="utf-8"):
        r = json.loads(line)
        if r.get("error"):
            continue

        qid = r["query_id"]
        cid = r["chunk_id"]
        j = r.get("llm_judgment", {})
        grade = int(j.get("relevance_grade", 0))

        if grade >= args.min_grade:
            if cid not in rel[qid]:
                rel[qid].append(cid)
            grades[qid][cid] = grade
            for p in j.get("answer_key_points", []) or []:
                if p and p not in key_points[qid]:
                    key_points[qid].append(p)

    with open(args.output, "w", encoding="utf-8") as f:
        for qid, tmpl in templates.items():
            x = dict(tmpl)
            x["id"] = qid
            x["relevant_chunk_ids"] = rel.get(qid, [])
            x["relevance_grades"] = grades.get(qid, {})
            x["answer_key_points"] = key_points.get(qid, [])[:12]

            if x["relevant_chunk_ids"]:
                x["gold_answer"] = (
                    "A good answer should cite the selected relevant evidence chunks and summarize: "
                    + "; ".join(x["answer_key_points"][:8])
                )
            else:
                x["gold_answer"] = "No relevant chunk was selected by LLM-assisted judging; manual review required."

            meta = dict(x.get("metadata") or {})
            meta["qrels_source"] = "llm_assisted_full_candidate_pool"
            meta["min_grade"] = args.min_grade
            meta["manual_review_required"] = True
            x["metadata"] = meta

            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print("wrote", args.output)
    print("queries", len(templates))
    print("with qrels", sum(1 for q in templates if rel.get(q)))

if __name__ == "__main__":
    main()
