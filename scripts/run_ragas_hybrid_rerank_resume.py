#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path

def load_jsonl(path):
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

def load_done_ids(output_jsonl):
    p = Path(output_jsonl)
    done = set()
    if not p.exists():
        return done

    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                x = json.loads(line)
                qid = x.get("query_id")
                if qid:
                    done.add(qid)
            except Exception:
                pass

    return done

def build_llm(timeout):
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    model = os.getenv("LLM_MODEL", "qwen3.7-plus")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

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
        name = name.strip()

        if name == "faithfulness":
            cls = getattr(rm, "Faithfulness", None)
            if cls is None:
                raise RuntimeError("Faithfulness metric not found.")
            metrics[name] = cls(llm=llm)

        elif name == "context_precision":
            cls = (
                getattr(rm, "LLMContextPrecisionWithReference", None)
                or getattr(rm, "ContextPrecision", None)
                or getattr(rm, "context_precision", None)
            )
            if cls is None:
                raise RuntimeError("Context precision metric not found.")
            metrics[name] = cls(llm=llm) if isinstance(cls, type) else cls

        elif name == "context_recall":
            cls = (
                getattr(rm, "LLMContextRecall", None)
                or getattr(rm, "ContextRecall", None)
                or getattr(rm, "context_recall", None)
            )
            if cls is None:
                raise RuntimeError("Context recall metric not found.")
            metrics[name] = cls(llm=llm) if isinstance(cls, type) else cls

        else:
            raise ValueError(f"Unsupported metric: {name}")

    return metrics

def build_sample(row, max_contexts, max_context_chars):
    from ragas.dataset_schema import SingleTurnSample

    qid = row.get("query_id") or row.get("id")
    question = row.get("question") or row.get("query") or ""
    answer = row.get("answer") or row.get("answer_preview") or ""
    reference = row.get("reference") or row.get("gold_answer") or ""

    contexts = row.get("contexts") or row.get("retrieved_contexts") or []
    contexts = [str(c)[:max_context_chars] for c in contexts[:max_contexts] if c]

    return qid, SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference,
    )

async def score_one(metric, sample, timeout):
    return await asyncio.wait_for(metric.single_turn_ascore(sample), timeout=timeout)

def summarize(output_jsonl, summary_json, summary_csv, per_query_csv):
    rows = []
    with open(output_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    metric_names = sorted({
        m
        for r in rows
        for m in (r.get("scores") or {}).keys()
    })

    summary = {
        "output_jsonl": output_jsonl,
        "num_evaluated_examples": len(rows),
        "summary_scores": {},
    }

    for m in metric_names:
        vals = [
            r["scores"].get(m)
            for r in rows
            if isinstance((r.get("scores") or {}).get(m), (int, float))
        ]
        summary["summary_scores"][m] = statistics.mean(vals) if vals else None
        summary["summary_scores"][f"{m}_valid_count"] = len(vals)
        summary["summary_scores"][f"{m}_failed_count"] = len(rows) - len(vals)

    Path(summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["num_evaluated_examples", len(rows)])
        for k, v in summary["summary_scores"].items():
            w.writerow([k, v])

    fields = ["query_id", "query"] + metric_names + [m + "_seconds" for m in metric_names] + ["failed_metrics"]
    with open(per_query_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {
                "query_id": r.get("query_id"),
                "query": r.get("query"),
                "failed_metrics": json.dumps(r.get("failed_metrics", {}), ensure_ascii=False),
            }
            for m in metric_names:
                out[m] = (r.get("scores") or {}).get(m)
                out[m + "_seconds"] = r.get(m + "_seconds")
            w.writerow(out)

    print("wrote", summary_json)
    print("wrote", summary_csv)
    print("wrote", per_query_csv)
    print(json.dumps(summary["summary_scores"], ensure_ascii=False, indent=2))

async def main_async(args):
    rows = load_jsonl(args.eval_results)
    if args.max_examples:
        rows = rows[:args.max_examples]

    done = load_done_ids(args.output_jsonl)
    print("total input rows:", len(rows))
    print("already done:", len(done))
    print("remaining:", len([r for r in rows if (r.get("query_id") or r.get("id")) not in done]))

    llm = build_llm(args.timeout)
    metrics = build_metrics(llm, [x.strip() for x in args.metrics.split(",") if x.strip()])

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_jsonl, "a", encoding="utf-8") as fout:
        for idx, row in enumerate(rows, 1):
            qid = row.get("query_id") or row.get("id")

            if qid in done:
                continue

            qid, sample = build_sample(row, args.max_contexts, args.max_context_chars)

            item = {
                "query_id": qid,
                "query": getattr(sample, "user_input", ""),
                "scores": {},
                "failed_metrics": {},
                "model": os.getenv("LLM_MODEL", "qwen3.7-plus"),
                "max_contexts": args.max_contexts,
                "max_context_chars": args.max_context_chars,
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
                    item[f"{metric_name}_seconds"] = round(time.perf_counter() - t0, 3)

                    print(f"  {metric_name}: FAILED {err}", flush=True)

                    if args.print_traceback:
                        traceback.print_exc()

                if args.sleep > 0:
                    await asyncio.sleep(args.sleep)

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()

    summarize(
        args.output_jsonl,
        args.summary_json,
        args.summary_csv,
        args.per_query_csv,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--per-query-csv", required=True)
    parser.add_argument("--metrics", default="faithfulness,context_precision,context_recall")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-contexts", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--print-traceback", action="store_true")
    args = parser.parse_args()

    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
