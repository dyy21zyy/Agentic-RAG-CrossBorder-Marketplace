#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CHUNK_ID_RE = re.compile(r"chunk_id:\s*([^\n\r]+)")


def load_chunks(path: Path) -> dict[str, dict[str, Any]]:
    chunks = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            cid = (
                r.get("chunk_id")
                or r.get("id")
                or r.get("_id")
                or (r.get("metadata") or {}).get("chunk_id")
            )

            if cid:
                chunks[str(cid)] = r

    return chunks


def parse_chunk_id(ctx: str) -> str | None:
    m = CHUNK_ID_RE.search(ctx or "")
    if m:
        return m.group(1).strip()

    # fallback
    for pattern in [
        r"(patent:\d+:patent_claim:claim-[^\s,\]\)']+)",
        r"(trademark:[^\s,\]\)']+)",
        r"(litigation:[^\s,\]\)']+)",
    ]:
        m = re.search(pattern, ctx or "")
        if m:
            return m.group(1).strip()

    return None


def get_source(r: dict[str, Any]) -> str:
    st = r.get("source_type") or (r.get("metadata") or {}).get("source_type") or ""
    ss = r.get("source_subtype") or (r.get("metadata") or {}).get("source_subtype") or ""

    if st and ss:
        return f"{st}/{ss}"
    return st or ss


def build_full_context(old_ctx: str, chunks: dict[str, dict[str, Any]], max_chars: int) -> tuple[str, bool]:
    cid = parse_chunk_id(old_ctx)

    if not cid or cid not in chunks:
        return old_ctx, False

    r = chunks[cid]

    title = r.get("title") or (r.get("metadata") or {}).get("title") or ""
    source = get_source(r)
    content = (
        r.get("content")
        or r.get("text")
        or r.get("page_content")
        or r.get("claim_text")
        or r.get("goods_services")
        or r.get("description")
        or r.get("long_description")
        or ""
    )

    content = str(content).strip()

    parts = [
        f"chunk_id: {cid}",
    ]

    if source:
        parts.append(f"source: {source}")

    if title:
        parts.append(f"title: {title}")

    if content:
        parts.append("content:")
        parts.append(content[:max_chars])

    return "\n".join(parts), bool(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-context-chars", type=int, default=1800)
    args = ap.parse_args()

    input_path = Path(args.input)
    chunks_path = Path(args.chunks)
    output_path = Path(args.output)

    print("loading chunks:", chunks_path)
    chunks = load_chunks(chunks_path)
    print("chunks loaded =", len(chunks))

    rows = 0
    context_count = 0
    enriched_count = 0
    before_chars = 0
    after_chars = 0

    with open(input_path, "r", encoding="utf-8") as f, open(output_path, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            old_contexts = r.get("retrieved_contexts") or r.get("contexts") or []
            new_contexts = []

            for ctx in old_contexts:
                ctx = str(ctx)

                context_count += 1
                before_chars += len(ctx)

                new_ctx, ok = build_full_context(
                    ctx,
                    chunks,
                    max_chars=args.max_context_chars,
                )

                after_chars += len(new_ctx)
                if ok:
                    enriched_count += 1

                new_contexts.append(new_ctx)

            r["retrieved_contexts"] = new_contexts
            r["contexts"] = new_contexts

            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            rows += 1

    print("input =", input_path)
    print("output =", output_path)
    print("rows =", rows)
    print("contexts =", context_count)
    print("enriched_contexts =", enriched_count)
    print("enriched_ratio =", round(enriched_count / context_count, 4) if context_count else 0)
    print("avg_context_chars_before =", round(before_chars / context_count, 2) if context_count else 0)
    print("avg_context_chars_after =", round(after_chars / context_count, 2) if context_count else 0)


if __name__ == "__main__":
    main()
