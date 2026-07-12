#!/usr/bin/env python
import json
from pathlib import Path
from types import SimpleNamespace

from crossborder_agentic_rag.retrieval.bm25 import LocalBM25Retriever, _extract_exact_ids

EVAL = Path("data/eval/chunk_grounded_eval_v1.jsonl")
CHUNKS = Path("data/processed/ip_evidence_chunks_full_optimized_fixed.jsonl")


def make_chunk(r):
    metadata = r.get("metadata") or {}

    return SimpleNamespace(
        chunk_id=r.get("chunk_id") or r.get("id") or "",
        doc_id=r.get("doc_id") or metadata.get("doc_id") or "",
        parent_id=r.get("parent_id") or metadata.get("parent_id") or "",
        title=r.get("title") or "",
        content=r.get("content") or r.get("text") or "",
        source_type=r.get("source_type") or "",
        source_subtype=r.get("source_subtype") or "",
        metadata=metadata,
    )


chunks = []
with open(CHUNKS, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(make_chunk(json.loads(line)))

print("loaded chunks =", len(chunks))

bm25 = LocalBM25Retriever(chunks)

total = 0
hit5 = 0

with open(EVAL, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        r = json.loads(line)
        q = r.get("query") or r.get("question") or r.get("input") or ""

        if "patent claim evidence for patent" not in q:
            continue

        golds = set(r.get("relevant_chunk_ids") or r.get("gold_chunk_ids") or r.get("qrels") or [])

        hits = bm25.search(
            q,
            filters=None,
            source_types=["patent"],
            top_k=5,
        )

        ids = [getattr(h, "chunk_id", "") for h in hits]
        scores = [getattr(h, "score", None) for h in hits]

        ok = bool(golds & set(ids))
        total += 1
        hit5 += int(ok)

        print("\nQUERY:", q)
        print("EXACT IDS:", _extract_exact_ids(q))
        print("GOLD:", list(golds))
        print("TOP5:", ids)
        print("SCORES:", scores)
        print("HIT:", ok)

print("\nTOTAL =", total)
print("BM25 patent exact Hit@5 =", hit5 / total if total else None)
