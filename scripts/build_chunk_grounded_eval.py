#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

def norm(s, n=100):
    return " ".join(str(s or "").split())[:n]

def make_query(c):
    st = c.get("source_type")
    sub = c.get("source_subtype")
    title = c.get("title") or ""
    md = c.get("metadata") or {}

    if st == "trademark":
        word = md.get("word_mark") or title
        if sub == "trademark_goods_services":
            return f"Find trademark goods and services evidence for {word}."
        if sub == "trademark_class":
            return f"Which Nice class evidence is associated with the trademark {word}?"
        return f"Find trademark identity or registration evidence for {word}."

    if st == "patent":
        patent_id = md.get("patent_id") or c.get("doc_id", "").replace("patent:", "")
        return f"Find patent claim evidence for patent {patent_id}."

    if st == "litigation":
        case_number = md.get("case_number")
        case_name = md.get("case_name") or title
        case_ref = case_number or case_name

        if sub == "litigation_docket":
            return f"Find litigation docket evidence for case {case_ref}."
        if sub == "litigation_party":
            return f"Find parties involved in the litigation case {case_ref}."
        if sub == "litigation_patent":
            return f"Find asserted patent evidence in litigation case {case_ref}."
        if sub == "litigation_timeline":
            return f"Find litigation timeline evidence for case {case_ref}."
        return f"Find litigation case summary evidence for {case_ref}."

    return f"Find evidence related to {norm(title)}."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--per-subtype", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    buckets = defaultdict(list)

    with open(args.chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            key = (c.get("source_type"), c.get("source_subtype"))
            if c.get("chunk_id") and c.get("content"):
                buckets[key].append(c)

    examples = []
    idx = 1

    for key in sorted(buckets):
        items = buckets[key]
        random.shuffle(items)

        for c in items[:args.per_subtype]:
            st, sub = key
            qid = f"CG{idx:04d}"
            cid = c["chunk_id"]

            ex = {
                "id": qid,
                "query": make_query(c),
                "query_type": st,
                "expected_route": "hybrid_retrieval",
                "expected_answer_type": f"{st}_answer",
                "expected_source_types": [st],
                "expected_source_subtypes": [sub],
                "relevant_doc_ids": [c.get("doc_id")],
                "relevant_chunk_ids": [cid],
                "relevance_grades": {cid: 3},
                "must_contain_any": [],
                "gold_answer": (
                    f"A good answer should retrieve and cite chunk {cid}, "
                    f"summarize evidence from {c.get('title')}, and avoid legal advice."
                ),
                "answer_key_points": [c.get("title")],
                "metadata": {
                    "source": "chunk_grounded_auto",
                    "source_type": st,
                    "source_subtype": sub,
                    "title": c.get("title"),
                },
            }
            examples.append(ex)
            idx += 1

    with open(args.output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("wrote", args.output)
    print("examples", len(examples))
    print("subtypes", len(buckets))

if __name__ == "__main__":
    main()
