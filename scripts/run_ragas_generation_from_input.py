#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            q = r.get("user_input") or r.get("question") or ""
            a = r.get("response") or r.get("answer") or ""
            ctx = r.get("retrieved_contexts") or r.get("contexts") or []
            ref = r.get("reference") or r.get("ground_truth") or ""

            if not q or not a or not ctx:
                continue

            rows.append({
                "id": r.get("id"),
                "user_input": q,
                "response": a,
                "retrieved_contexts": ctx,
                "reference": ref,

                # 兼容旧版 RAGAS
                "question": q,
                "answer": a,
                "contexts": ctx,
                "ground_truth": ref,

                "query_type": r.get("query_type"),
                "task_type": r.get("task_type"),
            })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=os.environ.get("RAGAS_LLM_MODEL") or os.environ.get("LLM_MODEL"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or "EMPTY")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(Path(args.input))
    if args.limit:
        rows = rows[:args.limit]

    print("valid rows =", len(rows))
    if not rows:
        raise SystemExit("No valid rows")

    dataset = Dataset.from_list(rows)

    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings

    lc_llm = ChatOpenAI(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=0,
        timeout=180,
        max_retries=3,
    )

    try:
        from ragas.llms import LangchainLLMWrapper
        evaluator_llm = LangchainLLMWrapper(lc_llm)
    except Exception:
        evaluator_llm = lc_llm

    emb_model = (
        os.environ.get("RAGAS_EMBEDDING_MODEL")
        or os.environ.get("RAG_EMBEDDING_MODEL")
        or os.environ.get("EMBEDDING_MODEL")
    )

    embeddings = None
    if emb_model:
        lc_embeddings = HuggingFaceEmbeddings(
            model_name=emb_model,
            model_kwargs={"device": os.environ.get("EMBEDDING_DEVICE", "cuda")},
            encode_kwargs={"normalize_embeddings": True},
        )

        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper
            embeddings = LangchainEmbeddingsWrapper(lc_embeddings)
        except Exception:
            embeddings = lc_embeddings

        print("embedding model =", emb_model)

    from ragas import evaluate

    try:
        from ragas.metrics import faithfulness, answer_relevancy
        metrics = [faithfulness, answer_relevancy]
    except Exception:
        from ragas.metrics import Faithfulness, ResponseRelevancy
        metrics = [Faithfulness(), ResponseRelevancy()]

    print("metrics =", [getattr(m, "name", str(m)) for m in metrics])
    print("llm model =", args.model)
    print("base_url =", args.base_url)

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()

    out_csv = out_dir / "ragas_scores_generation.csv"
    out_json = out_dir / "ragas_summary_generation.json"

    df.to_csv(out_csv, index=False)

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    summary = {
        "n": len(df),
        "metrics": {
            c: {
                "mean": float(df[c].mean()),
                "median": float(df[c].median()),
            }
            for c in numeric_cols
        },
        "input": str(args.input),
        "model": args.model,
        "base_url": args.base_url,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(out_csv)
    print(out_json)
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
