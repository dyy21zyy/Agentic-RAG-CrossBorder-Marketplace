#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
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


def build_chat_openai(model: str, api_key: str, base_url: str, timeout: int):
    from langchain_openai import ChatOpenAI

    common = dict(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=timeout,
        max_retries=2,
        n=1,
    )

    # 关键：禁用 Qwen thinking，避免 enable_thinking=true 时 n>1 报错
    try:
        return ChatOpenAI(
            **common,
            extra_body={"enable_thinking": False},
        )
    except TypeError:
        return ChatOpenAI(
            **common,
            model_kwargs={"extra_body": {"enable_thinking": False}},
        )


def wrap_llm(lc_llm):
    try:
        from ragas.llms import LangchainLLMWrapper
        return LangchainLLMWrapper(lc_llm)
    except Exception:
        return lc_llm


def build_embeddings():
    emb_model = (
        os.environ.get("RAGAS_EMBEDDING_MODEL")
        or os.environ.get("RAG_EMBEDDING_MODEL")
        or os.environ.get("EMBEDDING_MODEL")
    )

    if not emb_model:
        return None

    from langchain_huggingface import HuggingFaceEmbeddings

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
    return embeddings


def make_metrics(metric_names: str):
    wanted = [x.strip() for x in metric_names.split(",") if x.strip()]
    metrics = []

    for name in wanted:
        if name == "faithfulness":
            try:
                from ragas.metrics import Faithfulness
                m = Faithfulness()
            except Exception:
                from ragas.metrics import faithfulness
                m = faithfulness

            metrics.append(m)

        elif name in {"answer_relevancy", "answer_relevance", "response_relevancy"}:
            # 关键：强制 strictness=1，避免 RAGAS answer_relevancy 请求 n=3
            try:
                from ragas.metrics import ResponseRelevancy
                try:
                    m = ResponseRelevancy(strictness=1)
                except TypeError:
                    m = ResponseRelevancy()
                    if hasattr(m, "strictness"):
                        m.strictness = 1
            except Exception:
                from ragas.metrics import answer_relevancy
                m = answer_relevancy
                if hasattr(m, "strictness"):
                    m.strictness = 1

            metrics.append(m)

        else:
            raise ValueError(f"Unsupported metric: {name}")

    return metrics


def make_run_config(timeout: int, max_workers: int, max_retries: int):
    try:
        from ragas.run_config import RunConfig
        kwargs = {}

        sig = inspect.signature(RunConfig)
        if "timeout" in sig.parameters:
            kwargs["timeout"] = timeout
        if "max_workers" in sig.parameters:
            kwargs["max_workers"] = max_workers
        if "max_retries" in sig.parameters:
            kwargs["max_retries"] = max_retries

        return RunConfig(**kwargs)
    except Exception as e:
        print("RunConfig unavailable, fallback without run_config:", repr(e))
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--metrics", default="faithfulness,answer_relevancy")
    ap.add_argument("--model", default=os.environ.get("RAGAS_LLM_MODEL") or os.environ.get("LLM_MODEL"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or "EMPTY")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--max-retries", type=int, default=2)
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

    lc_llm = build_chat_openai(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    evaluator_llm = wrap_llm(lc_llm)
    embeddings = build_embeddings()
    metrics = make_metrics(args.metrics)
    run_config = make_run_config(
        timeout=args.timeout,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )

    print("metrics =", [getattr(m, "name", type(m).__name__) for m in metrics])
    print("llm model =", args.model)
    print("base_url =", args.base_url)
    print("disable_thinking = True")
    print("answer_relevancy strictness = 1")
    print("timeout =", args.timeout)
    print("max_workers =", args.max_workers)
    print("max_retries =", args.max_retries)

    from ragas import evaluate

    eval_kwargs = dict(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=embeddings,
    )

    sig = inspect.signature(evaluate)
    if run_config is not None and "run_config" in sig.parameters:
        eval_kwargs["run_config"] = run_config

    result = evaluate(**eval_kwargs)
    df = result.to_pandas()

    metric_tag = args.metrics.replace(",", "_").replace(" ", "")
    if args.limit:
        metric_tag += f"_limit{args.limit}"

    out_csv = out_dir / f"ragas_scores_{metric_tag}.csv"
    out_json = out_dir / f"ragas_summary_{metric_tag}.json"

    df.to_csv(out_csv, index=False)

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    summary = {
        "n": len(df),
        "metrics_requested": args.metrics,
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
        "disable_thinking": True,
        "answer_relevancy_strictness": 1,
        "timeout": args.timeout,
        "max_workers": args.max_workers,
        "max_retries": args.max_retries,
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
