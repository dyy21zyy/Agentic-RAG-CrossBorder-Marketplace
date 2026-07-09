#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

def load_jsonl(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def clean_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def get_value(result):
    if hasattr(result, "value"):
        return clean_float(result.value)
    return clean_float(result)

def build_llm(timeout: int):
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    model = os.getenv("LLM_MODEL", "qwen3.7-plus")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is missing")

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            timeout=timeout,
            max_retries=1,
        )
    )

def build_metrics(llm, metric_names):
    import ragas.metrics as rm

    metrics = {}

    for name in metric_names:
        if name == "faithfulness":
            cls = getattr(rm, "Faithfulness", None)
            if cls is None:
                raise RuntimeError("Faithfulness metric not found in ragas.metrics")
            metrics[name] = cls(llm=llm)

        elif name == "context_precision":
            cls = (
                getattr(rm, "LLMContextPrecisionWithReference", None)
                or getattr(rm, "ContextPrecision", None)
                or getattr(rm, "context_precision", None)
            )
            if cls is None:
                raise RuntimeError("Context precision metric not found in ragas.metrics")
            if isinstance(cls, type):
                metrics[name] = cls(llm=llm)
            else:
                metrics[name] = cls

        elif name == "context_recall":
            cls = (
                getattr(rm, "LLMContextRecall", None)
                or getattr(rm, "ContextRecall", None)
                or getattr(rm, "context_recall", None)
            )
            if cls is None:
                raise RuntimeError("Context recall metric not found in ragas.metrics")
            if isinstance(cls, type):
                metrics[name] = cls(llm=llm)
            else:
                metrics[name] = cls

        else:
            raise ValueError(f"Unsupported metric for stable runner: {name}")

    return metrics

def build_sample(row, max_contexts: int, max_context_chars: int):
    from ragas.dataset_schema import SingleTurnSample

    qid = row.get("query_id") or row.get("id")
    question = row.get("question") or row.get("query") or ""
    answer = row.get("answer") or row.get("answer_preview") or ""
    reference = row.get("reference") or row.get("gold_answer") or row.get("ground_truth") or ""

    contexts = row.get("contexts") or row.get("retrieved_contexts") or []
    contexts = [str(c)[:max_context_chars] for c in contexts[:max_contexts] if c]

    return qid, SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference,
    )

async def score_one(metric, sample, timeout: int):
    return await asyncio.wait_for(metric.single_turn_ascore(sample), timeout=timeout)

async def main_async(args):
    rows = load_jsonl(args.eval_results)
    if args.max_examples:
        rows = rows[: args.max_examples]

    llm = build_llm(args.timeout)
    metrics = build_metrics(llm, args.metrics.split(","))

    per_query = []
    failures = []

    for idx, row in enumerate(rows, 1):
        qid, sample = build_sample(row, args.max_contexts, args.max_context_chars)

        item = {
            "query_id": qid,
            "query": getattr(sample, "user_input", ""),
            "scores": {},
            "failed_metrics": {},
        }

        print(f"[{idx}/{len(rows)}] scoring {qid}", flush=True)

        for metric_name, metric in metrics.items():
            t0 = time.perf_counter()
            try:
                result = await score_one(metric, sample, args.timeout)
                value = get_value(result)
                item["scores"][metric_name] = value
                item[f"{metric_name}_seconds"] = round(time.perf_counter() - t0, 3)

                if value is None:
                    item["failed_metrics"][metric_name] = "nan_or_invalid"

                print(f"  {metric_name}: {value}", flush=True)

            except Exception as exc:
                err = repr(exc)
                item["scores"][metric_name] = None
                item["failed_metrics"][metric_name] = err
                failures.append({"query_id": qid, "metric": metric_name, "error": err})
                print(f"  {metric_name}: FAILED {err}", flush=True)

                if args.raise_exceptions:
                    traceback.print_exc()
                    raise

            if args.sleep > 0:
                await asyncio.sleep(args.sleep)

        per_query.append(item)

    summary = {}
    for metric_name in metrics:
        vals = [
            x["scores"].get(metric_name)
            for x in per_query
            if isinstance(x["scores"].get(metric_name), (int, float))
        ]
        summary[metric_name] = statistics.mean(vals) if vals else None
        summary[f"{metric_name}_valid_count"] = len(vals)
        summary[f"{metric_name}_failed_count"] = len(per_query) - len(vals)

    report = {
        "input": args.eval_results,
        "model": os.getenv("LLM_MODEL", "qwen3.7-plus"),
        "num_input_examples": len(rows),
        "metrics": list(metrics.keys()),
        "max_contexts": args.max_contexts,
        "max_context_chars": args.max_context_chars,
        "timeout": args.timeout,
        "summary_scores": summary,
        "per_query_scores": per_query,
        "failures": failures,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote", out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="faithfulness,context_precision,context_recall")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-contexts", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=600)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--raise-exceptions", action="store_true")
    args = parser.parse_args()

    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
