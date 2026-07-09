#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

def run_query(query: str, args):
    cmd = [
        sys.executable,
        "scripts/08_run_query_cli.py",
        query,
        "--duckdb-path", args.duckdb_path,
        "--collection-name", args.collection_name,
        "--embedding-provider", args.embedding_provider,
        "--retrieval-mode", args.retrieval_mode,
        "--top-k", str(args.top_k),
        "--output-json",
    ]

    if args.use_milvus:
        cmd.append("--use-milvus")

    if args.chunks_path:
        cmd.extend(["--chunks-path", args.chunks_path])

    st = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    latency_ms = (time.perf_counter() - st) * 1000

    if p.returncode != 0:
        return {
            "ok": False,
            "error": p.stderr.strip() or p.stdout.strip(),
            "latency_ms": latency_ms,
        }

    try:
        data = json.loads(p.stdout)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"JSON parse failed: {exc}; stdout={p.stdout[:500]}; stderr={p.stderr[:500]}",
            "latency_ms": latency_ms,
        }

    data["ok"] = True
    data["latency_ms"] = latency_ms
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--duckdb-path", default="data/processed/ip_structured.duckdb")
    ap.add_argument("--collection-name", default="ip_rag_collection")
    ap.add_argument("--embedding-provider", default="local")
    ap.add_argument("--retrieval-mode", default="dense_only")
    ap.add_argument("--chunks-path")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--use-milvus", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in open(ROOT / args.eval_path, encoding="utf-8"):
        if not line.strip():
            continue

        ex = json.loads(line)
        qid = ex.get("id") or ex.get("query_id")
        query = ex["query"]

        print(f"Running {qid}: {query}", flush=True)
        res = run_query(query, args)

        chunks = res.get("source_chunks", []) if res.get("ok") else []
        source_counts = Counter(c.get("source_type") for c in chunks)
        subtype_counts = Counter(c.get("source_subtype") for c in chunks)

        row = {
            "query_id": qid,
            "query": query,
            "template": ex,
            "ok": res.get("ok", False),
            "error": res.get("error"),
            "latency_ms": res.get("latency_ms"),
            "query_type": res.get("query_type"),
            "retrieval_route": res.get("retrieval_route"),
            "source_type_counts": dict(source_counts),
            "source_subtype_counts": dict(subtype_counts),
            "candidate_chunks": chunks,
            "answer_preview": res.get("answer", "")[:1000] if res.get("ok") else "",
        }
        rows.append(row)

    with open(out_dir / "candidates.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "n": len(rows),
        "ok": sum(1 for r in rows if r["ok"]),
        "failed": sum(1 for r in rows if not r["ok"]),
        "avg_latency_ms": sum(r["latency_ms"] for r in rows if r["latency_ms"] is not None) / max(1, len(rows)),
    }

    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Wrote:", out_dir / "candidates.jsonl")

if __name__ == "__main__":
    main()
