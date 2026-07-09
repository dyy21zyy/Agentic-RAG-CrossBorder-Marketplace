#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path
from collections import Counter

def file_info(path: str):
    p = Path(path)
    if not p.exists():
        return {"path": path, "exists": False}
    return {
        "path": path,
        "exists": True,
        "size_bytes": p.stat().st_size,
        "size_mb": round(p.stat().st_size / 1024 / 1024, 3),
    }

def count_lines(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))

def main():
    manifest = {}

    files = {
        "normalized_docs_all": "data/processed/normalized_docs_all.jsonl",
        "chunks_full_fixed": "data/processed/ip_evidence_chunks_full_optimized_fixed.jsonl",
        "duckdb": "data/processed/ip_structured.duckdb",
        "eval_file": "data/eval/ip_eval_v1_full.jsonl",
        "golden_queries_30": "data/eval/golden_queries_30.jsonl",
        "golden_queries_30_filled": "data/eval/golden_queries_30_filled_llm.jsonl",
        "chunk_grounded_eval": "data/eval/chunk_grounded_eval_v1.jsonl",
    }

    manifest["files"] = {k: file_info(v) for k, v in files.items()}
    manifest["line_counts"] = {k: count_lines(v) for k, v in files.items() if v.endswith(".jsonl")}

    chunks_path = Path(files["chunks_full_fixed"])
    if chunks_path.exists():
        source = Counter()
        subtype = Counter()
        partition = Counter()
        seen = set()
        dups = 0

        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                cid = x.get("chunk_id")
                if cid in seen:
                    dups += 1
                seen.add(cid)
                source[x.get("source_type")] += 1
                subtype[x.get("source_subtype")] += 1
                md = x.get("metadata") or {}
                partition[md.get("partition") or x.get("partition")] += 1

        manifest["chunks"] = {
            "total": sum(source.values()),
            "unique_chunk_ids": len(seen),
            "duplicate_chunk_ids": dups,
            "source_type_counts": dict(source),
            "source_subtype_counts": dict(subtype),
            "partition_counts": dict(partition),
        }

    eval_path = Path(files["eval_file"])
    if eval_path.exists():
        qtypes = Counter()
        expected_types = Counter()
        expected_subtypes = Counter()
        with_qrels = 0
        total = 0

        with eval_path.open("r", encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                total += 1
                qtypes[x.get("query_type")] += 1
                if x.get("relevant_chunk_ids"):
                    with_qrels += 1
                for t in x.get("expected_source_types", []):
                    expected_types[t] += 1
                for s in x.get("expected_source_subtypes", []):
                    expected_subtypes[s] += 1

        manifest["eval_dataset"] = {
            "total": total,
            "with_qrels": with_qrels,
            "query_type_counts": dict(qtypes),
            "expected_source_type_counts": dict(expected_types),
            "expected_source_subtype_counts": dict(expected_subtypes),
        }

    # Milvus row count
    try:
        from pymilvus import MilvusClient

        env = {}
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")

        uri = env.get("RAG_MILVUS_URI") or env.get("MILVUS_URI")
        collection = env.get("MILVUS_COLLECTION_NAME", "ip_rag_collection")

        if uri:
            client = MilvusClient(uri=uri)
            client.load_collection(collection_name=collection)
            stats = client.get_collection_stats(collection_name=collection)
            manifest["milvus"] = {
                "uri": uri,
                "collection": collection,
                "stats": stats,
                "row_count": int(stats.get("row_count", 0)),
            }
    except Exception as e:
        manifest["milvus"] = {"error": repr(e)}

    out = Path("reports/eval_full_v1_timed/offline_artifact_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("wrote", out)

if __name__ == "__main__":
    main()
