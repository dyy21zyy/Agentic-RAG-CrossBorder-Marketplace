#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inp = Path(args.comparison)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ok = 0
    bad = 0

    with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            q = r.get("ragas_user_input") or r.get("query") or ""
            response = r.get("ragas_response") or r.get("answer_preview") or ""
            contexts = r.get("ragas_retrieved_contexts") or []
            reference = r.get("ragas_reference") or ""

            if not q or not response or not contexts:
                bad += 1
                continue

            out_row = {
                "id": r.get("id") or r.get("idx"),
                "user_input": q,
                "response": response,
                "retrieved_contexts": contexts,
                "reference": reference,

                "question": q,
                "answer": response,
                "contexts": contexts,
                "ground_truth": reference,

                "query_type": r.get("query_type"),
                "task_type": r.get("task_type"),
                "latency_ms": r.get("latency_ms"),
                "retrieval_ms": r.get("retrieval_ms"),
                "precision_at_5": r.get("precision_at_5"),
                "recall_at_5": r.get("recall_at_5"),
                "hit_at_5": r.get("hit_at_5"),
                "mrr_at_5": r.get("mrr_at_5"),
                "ndcg_at_5": r.get("ndcg_at_5"),
                "map_at_5": r.get("map_at_5"),
            }

            g.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            ok += 1

    print("comparison =", inp)
    print("out =", out)
    print("ok =", ok)
    print("bad =", bad)


if __name__ == "__main__":
    main()
