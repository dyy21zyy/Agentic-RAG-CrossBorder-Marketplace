#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_gold_map(eval_path: Path):
    gold_map = {}

    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)
            qid = x.get("id") or x.get("query_id")

            if qid:
                gold_map[qid] = x

    return gold_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skipped-output", required=True)
    parser.add_argument("--max-contexts", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=1000)
    parser.add_argument("--max-answer-chars", type=int, default=1200)
    parser.add_argument("--strict-filter", action="store_true")
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    results_path = Path(args.results_path)
    output_path = Path(args.output)
    skipped_path = Path(args.skipped_output)

    if not eval_path.exists():
        raise FileNotFoundError(f"eval file not found: {eval_path}")

    if not results_path.exists():
        raise FileNotFoundError(f"results file not found: {results_path}")

    gold_map = load_gold_map(eval_path)

    # strict_filter=True 时，会过滤更多无效 reference；
    # 默认只过滤明确没有 relevant chunk 的样本，避免误删太多。
    if args.strict_filter:
        bad_phrases = [
            "No relevant chunk was selected",
            "no relevant chunk was selected",
            "No relevant evidence was selected",
            "manual review required",
            "Manual review required",
        ]
    else:
        bad_phrases = [
            "No relevant chunk was selected",
            "no relevant chunk was selected",
            "No relevant evidence was selected",
        ]

    rows = []
    skipped = []

    with open(results_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            qid = x.get("id") or x.get("query_id")
            g = gold_map.get(qid, {})

            query = x.get("query") or x.get("question") or g.get("query") or ""

            answer = x.get("answer") or x.get("answer_preview") or ""

            gold_answer = (
                x.get("gold_answer")
                or x.get("reference")
                or g.get("gold_answer")
                or g.get("reference")
                or ""
            )

            if not gold_answer:
                skipped.append({
                    "id": qid,
                    "reason": "missing_gold_answer",
                })
                continue

            if any(p in gold_answer for p in bad_phrases):
                skipped.append({
                    "id": qid,
                    "reason": "placeholder_or_invalid_gold_answer",
                    "gold_answer": gold_answer[:300],
                })
                continue

            contexts = x.get("retrieved_contexts") or x.get("contexts") or []

            if not contexts:
                contexts = []

                for h in x.get("hits", []) or []:
                    text = h.get("content") or h.get("content_preview") or ""
                    title = h.get("title") or ""
                    chunk_id = h.get("chunk_id") or ""
                    source_type = h.get("source_type") or ""
                    source_subtype = h.get("source_subtype") or ""

                    if text:
                        ctx = (
                            f"[chunk_id={chunk_id} "
                            f"source_type={source_type} "
                            f"source_subtype={source_subtype} "
                            f"title={title}]\n{text}"
                        )
                        contexts.append(ctx)

            contexts = [
                str(c)[: args.max_context_chars]
                for c in contexts[: args.max_contexts]
                if c
            ]

            if not answer:
                skipped.append({
                    "id": qid,
                    "reason": "missing_answer",
                })
                continue

            if not contexts:
                skipped.append({
                    "id": qid,
                    "reason": "missing_contexts",
                })
                continue

            rows.append({
                "id": qid,
                "query_id": qid,
                "query": query,
                "question": query,
                "answer": answer[: args.max_answer_chars],
                "contexts": contexts,
                "retrieved_contexts": contexts,
                "gold_answer": gold_answer,
                "reference": gold_answer,
                "mode": "hybrid_rerank",
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("eval_path:", eval_path)
    print("results_path:", results_path)
    print("output:", output_path)
    print("kept:", len(rows))
    print("skipped:", len(skipped))
    print("skipped_output:", skipped_path)


if __name__ == "__main__":
    main()
