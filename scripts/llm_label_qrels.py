#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from openai import OpenAI

SYSTEM = """You are a strict relevance judge for a RAG evaluation dataset.
Judge whether each candidate evidence chunk is relevant to the user query.

Use only the query and chunk content. Do not use external knowledge.

Relevance grade:
3 = highly relevant, directly supports answering the query
2 = partially relevant, useful supporting evidence
1 = weakly related, same broad topic but not enough to answer
0 = not relevant

Return valid JSON only.
"""

def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text

def judge(client, model, query, expected_source_types, expected_source_subtypes, chunk):
    prompt = {
        "query": query,
        "expected_source_types": expected_source_types,
        "expected_source_subtypes": expected_source_subtypes,
        "candidate_chunk": {
            "chunk_id": chunk.get("chunk_id"),
            "source_type": chunk.get("source_type"),
            "source_subtype": chunk.get("source_subtype"),
            "title": chunk.get("title"),
            "content_preview": chunk.get("content_preview") or chunk.get("content", "")[:1600],
        },
        "output_schema": {
            "chunk_id": "string",
            "relevance_grade": "integer 0/1/2/3",
            "reason": "short reason",
            "answer_key_points": ["key points supported by this chunk"]
        }
    }

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
        ],
    )

    text = clean_json_text(resp.choices[0].message.content)
    return json.loads(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=os.getenv("LLM_MODEL"))
    ap.add_argument("--max-candidates-per-query", type=int, default=80)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    if not args.model:
        raise SystemExit("Missing --model or LLM_MODEL in env.")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["query_id"], r["chunk_id"]))
            except Exception:
                pass

    fout = open(out_path, "a", encoding="utf-8")

    for line in open(args.candidates, encoding="utf-8"):
        row = json.loads(line)
        qid = row["query_id"]
        query = row["query"]
        tmpl = row["template"]
        expected_source_types = tmpl.get("expected_source_types", [])
        expected_source_subtypes = tmpl.get("expected_source_subtypes", [])

        chunks = row.get("candidate_chunks", [])[:args.max_candidates_per_query]

        for c in chunks:
            cid = c.get("chunk_id")
            if not cid or (qid, cid) in done:
                continue

            try:
                j = judge(client, args.model, query, expected_source_types, expected_source_subtypes, c)
                result = {
                    "query_id": qid,
                    "query": query,
                    "chunk_id": cid,
                    "source_type": c.get("source_type"),
                    "source_subtype": c.get("source_subtype"),
                    "title": c.get("title"),
                    "candidate_from": c.get("candidate_from"),
                    "candidate_rank": c.get("candidate_rank"),
                    "llm_judgment": j,
                }
                grade = j.get("relevance_grade")
            except Exception as e:
                result = {
                    "query_id": qid,
                    "query": query,
                    "chunk_id": cid,
                    "error": repr(e),
                }
                grade = "ERR"

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            print(qid, cid, "grade=", grade)
            time.sleep(args.sleep)

    fout.close()

if __name__ == "__main__":
    main()
