#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-candidates", type=int, default=120)
    args = ap.parse_args()

    by_qid = {}

    for path in args.inputs:
        mode_name = Path(path).parent.name
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            qid = row["query_id"]

            if qid not in by_qid:
                by_qid[qid] = {
                    "query_id": qid,
                    "query": row["query"],
                    "template": row["template"],
                    "candidate_chunks": [],
                    "seen": set(),
                }

            for rank, c in enumerate(row.get("candidate_chunks", []), 1):
                cid = c.get("chunk_id")
                if not cid or cid in by_qid[qid]["seen"]:
                    continue
                c = dict(c)
                c["candidate_from"] = mode_name
                c["candidate_rank"] = rank
                by_qid[qid]["candidate_chunks"].append(c)
                by_qid[qid]["seen"].add(cid)

    rows = []
    for _, row in by_qid.items():
        row["candidate_chunks"] = row["candidate_chunks"][:args.max_candidates]
        row.pop("seen")
        rows.append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("wrote", args.output)
    print("queries", len(rows))

if __name__ == "__main__":
    main()
