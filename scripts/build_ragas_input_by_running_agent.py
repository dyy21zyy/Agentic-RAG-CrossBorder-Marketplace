#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("empty stdout")

    # 先尝试整体 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 再尝试截取最后一个 JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("cannot parse JSON from stdout")


def evidence_to_context(e: dict) -> str:
    title = e.get("title") or e.get("metadata", {}).get("title") or ""
    content = (
        e.get("content")
        or e.get("text")
        or e.get("page_content")
        or e.get("snippet")
        or ""
    )
    cid = e.get("chunk_id") or e.get("id") or ""
    source_type = e.get("source_type") or e.get("metadata", {}).get("source_type") or ""
    source_subtype = e.get("source_subtype") or e.get("metadata", {}).get("source_subtype") or ""

    parts = []
    if cid:
        parts.append(f"chunk_id: {cid}")
    if source_type or source_subtype:
        parts.append(f"source: {source_type}/{source_subtype}")
    if title:
        parts.append(f"title: {title}")
    if content:
        parts.append(content)

    return "\n".join(parts).strip()


def get_answer(d: dict) -> str:
    """Extract final answer from run_agentic_rag.py output.

    Current output uses:
    - llm_answer
    - deterministic_answer

    Prefer llm_answer when available; fallback to deterministic_answer.
    """
    for key in [
        "llm_answer",
        "deterministic_answer",
        "answer",
        "response",
        "final_answer",
        "answer_text",
        "final_response",
        "output_text",
        "answer_preview",
    ]:
        v = d.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Nested fallback, but avoid accidentally using evidence content as answer.
    for container_key in ["result", "output", "message"]:
        v = d.get(container_key)
        if isinstance(v, dict):
            for key in ["llm_answer", "deterministic_answer", "answer", "response", "final_answer", "content"]:
                vv = v.get(key)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()

    return ""


def get_evidence_list(d: dict) -> list[dict]:
    for key in [
        "final_evidence",
        "reranked_evidence",
        "retrieved_evidence",
        "sources",
        "evidence",
    ]:
        xs = d.get(key)
        if isinstance(xs, list) and xs:
            return xs
    return []


def build_reference(row: dict) -> str:
    parts = []

    if row.get("gold_answer"):
        parts.append(str(row["gold_answer"]))

    key_points = row.get("gold_answer_key_points") or row.get("answer_key_points") or []
    if key_points:
        parts.append("Key points: " + "; ".join(map(str, key_points)))

    target_entities = row.get("target_entities") or {}
    if target_entities:
        parts.append("Target entities: " + json.dumps(target_entities, ensure_ascii=False))

    gold_ids = row.get("relevant_chunk_ids") or []
    if gold_ids:
        parts.append("Relevant chunk ids: " + "; ".join(gold_ids[:10]))

    return "\n".join(parts).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    eval_path = Path(args.eval)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(load_jsonl(eval_path))
    if args.start:
        rows = rows[args.start:]
    if args.limit:
        rows = rows[:args.limit]

    print(f"Loaded eval rows: {len(rows)}")
    print(f"Saving RAGAS input to: {out_path}")

    base_cmd = [
        sys.executable,
        "scripts/run_agentic_rag.py",
        "--pipeline-mode", "agentic_llm",
        "--chunks-path", os.environ.get("CHUNKS_PATH", "data/processed/ip_evidence_chunks_full_optimized_fixed.jsonl"),
        "--duckdb-path", os.environ.get("DUCKDB_PATH", "data/processed/ip_structured.duckdb"),
        "--use-milvus",
        "--collection-name", os.environ.get("MILVUS_COLLECTION_NAME", "ip_rag_collection"),
        "--embedding-provider", os.environ.get("EMBEDDING_PROVIDER", "local"),
        "--retrieval-mode", "hybrid_rerank",
        "--reranker-provider", os.environ.get("RERANKER_PROVIDER", "local"),
        "--reranker-model", os.environ.get("RERANKER_MODEL", "/root/autodl-tmp/models/bge-reranker-base"),
        "--candidate-k", "20",
        "--top-k", "5",
        "--max-iterations", "3",
        "--use-llm",
        "--llm-provider", os.environ.get("LLM_PROVIDER", "openai"),
        "--llm-model", os.environ.get("LLM_MODEL", ""),
        "--llm-base-url", os.environ.get("OPENAI_BASE_URL", ""),
        "--output-json",
        "--show-trace",
        "--show-sources",
    ]

    ok = 0
    fail = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            q = row.get("query") or row.get("question") or row.get("user_input")
            if not q:
                continue

            print(f"[{i}/{len(rows)}] {row.get('id')} | {q[:120]}")

            cmd = base_cmd + ["--query", q]

            try:
                proc = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,
                    timeout=300,
                    check=False,
                )

                if proc.returncode != 0:
                    fail += 1
                    print("FAILED returncode:", proc.returncode)
                    print(proc.stderr[-2000:])
                    continue

                d = extract_json_object(proc.stdout)
                answer = get_answer(d)
                evs = get_evidence_list(d)
                contexts = [evidence_to_context(e) for e in evs]
                contexts = [x for x in contexts if x]

                if not answer:
                    print("WARN: empty answer")
                if not contexts:
                    print("WARN: empty contexts")

                out = {
                    "id": row.get("id"),
                    "user_input": q,
                    "response": answer,
                    "retrieved_contexts": contexts,
                    "reference": build_reference(row),

                    # 兼容旧版 ragas / datasets 字段
                    "question": q,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": build_reference(row),

                    # 方便分组分析
                    "query_type": row.get("query_type"),
                    "task_type": row.get("task_type"),
                    "expected_source_types": row.get("expected_source_types"),
                    "expected_source_subtypes": row.get("expected_source_subtypes"),
                    "relevant_chunk_ids": row.get("relevant_chunk_ids"),
                    "strict_relevant_chunk_ids": row.get("strict_relevant_chunk_ids"),
                    "target_entities": row.get("target_entities"),
                    "latency_ms": d.get("latency_ms"),
                    "trace": d.get("trace"),
                }

                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                ok += 1

            except Exception as e:
                fail += 1
                print("FAILED exception:", repr(e))

    print(f"Done. ok={ok}, fail={fail}, out={out_path}")


if __name__ == "__main__":
    main()
