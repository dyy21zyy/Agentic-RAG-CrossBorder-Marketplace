#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

def load_gold(eval_path):
    gold = {}
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            qid = x.get("id") or x.get("query_id")
            gold[qid] = x
    return gold

def get_contexts(x, max_context_chars=1600):
    contexts = x.get("retrieved_contexts") or x.get("contexts") or x.get("reference_contexts")
    if contexts:
        return [str(c)[:max_context_chars] for c in contexts if c]

    out = []
    for h in x.get("hits", []) or []:
        text = h.get("content") or h.get("content_preview") or ""
        title = h.get("title") or ""
        source_type = h.get("source_type") or ""
        source_subtype = h.get("source_subtype") or ""
        chunk_id = h.get("chunk_id") or ""

        ctx = f"[chunk_id={chunk_id} source_type={source_type} source_subtype={source_subtype} title={title}]\n{text}"
        if text:
            out.append(ctx[:max_context_chars])

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-path", required=True)
    ap.add_argument("--results-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-context-chars", type=int, default=1600)
    ap.add_argument("--require-answer", action="store_true")
    args = ap.parse_args()

    gold = load_gold(args.eval_path)
    rows = []
    skipped = []

    with open(args.results_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)
            qid = x.get("id") or x.get("query_id")
            g = gold.get(qid, {})

            query = x.get("query") or g.get("query") or ""
            answer = x.get("answer") or x.get("answer_preview") or ""
            contexts = get_contexts(x, args.max_context_chars)
            gold_answer = x.get("gold_answer") or g.get("gold_answer") or ""

            if args.require_answer and not answer:
                skipped.append({"id": qid, "reason": "missing answer"})
                continue

            if not contexts:
                skipped.append({"id": qid, "reason": "missing contexts"})
                continue

            rows.append({
                "id": qid,
                "query_id": qid,
                "query": query,
                "question": query,
                "answer": answer,
                "retrieved_contexts": contexts,
                "contexts": contexts,
                "gold_answer": gold_answer,
                "reference": gold_answer,
                "mode": x.get("mode"),
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    skipped_path = out.with_suffix(".skipped.json")
    skipped_path.write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    print("input =", args.results_path)
    print("output =", out)
    print("records =", len(rows))
    print("skipped =", len(skipped))
    print("skipped_path =", skipped_path)

if __name__ == "__main__":
    main()
