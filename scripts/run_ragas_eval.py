#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

DEFAULT_METRICS = "faithfulness,answer_relevancy,context_precision,context_recall"

METRIC_ALIASES = {
    "faithfulness": ["faithfulness", "Faithfulness"],
    "answer_relevancy": ["answer_relevancy", "answer_relevance", "AnswerRelevancy", "ResponseRelevancy"],
    "context_precision": [
        "context_precision",
        "LLMContextPrecisionWithoutReference",
        "LLMContextPrecisionWithReference",
        "ContextPrecision",
    ],
    "context_recall": ["context_recall", "LLMContextRecall", "ContextRecall"],
}

REFERENCE_METRICS = {
    "context_recall",
    "answer_correctness",
    "answer_similarity",
}

def load_eval_results(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing eval results file: {p}")

    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object on line {line_no}")
                rows.append(obj)
        return rows

    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]

    raise ValueError('Expected JSON list, JSONL, or JSON object with "results" list')

def _as_contexts(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(x) for x in value if x]
    return []

def build_ragas_records(items: list[dict[str, Any]], max_examples: int | None = None, require_contexts: bool = False):
    records = []
    skipped = []

    source_items = items[:max_examples] if max_examples is not None else items

    for idx, item in enumerate(source_items):
        query_id = str(item.get("query_id") or item.get("id") or idx)

        contexts = _as_contexts(
            item.get("retrieved_contexts")
            or item.get("contexts")
            or item.get("reference_contexts")
            or []
        )

        if not contexts:
            msg = {"query_id": query_id, "reason": "missing contexts"}
            if require_contexts:
                raise ValueError(f"missing contexts for query_id={query_id}")
            skipped.append(msg)
            continue

        question = str(item.get("query") or item.get("question") or "")
        answer = str(item.get("answer") or item.get("answer_preview") or "")
        ground_truth = str(item.get("gold_answer") or item.get("reference") or "")

        records.append({
            "query_id": query_id,
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "reference": ground_truth,
        })

    return records, skipped

def _parse_metrics(metrics: str) -> list[str]:
    return [m.strip() for m in metrics.split(",") if m.strip()]

def _instantiate_metric(obj: Any):
    # old ragas: metric is already an object
    # new ragas: metric may be a class
    try:
        if isinstance(obj, type):
            return obj()
        return obj
    except Exception:
        return obj

def _load_metric_objects(metric_names: list[str], has_gold: bool):
    skipped = {}
    selected = []
    used = []

    try:
        import ragas.metrics as ragas_metrics
    except Exception as exc:
        print("Failed to import ragas.metrics. This is not necessarily an installation problem.", file=sys.stderr)
        print("Actual exception:", repr(exc), file=sys.stderr)
        traceback.print_exc()
        raise

    for requested in metric_names:
        if requested in REFERENCE_METRICS and not has_gold:
            skipped[requested] = "Skipped because no gold_answer / reference is available"
            continue

        candidates = METRIC_ALIASES.get(requested, [requested])
        metric_obj = None
        metric_name_found = None

        for name in candidates:
            if hasattr(ragas_metrics, name):
                metric_obj = getattr(ragas_metrics, name)
                metric_name_found = name
                break

        if metric_obj is None:
            skipped[requested] = f"Metric not available in installed ragas version. Tried: {candidates}"
            continue

        metric_obj = _instantiate_metric(metric_obj)
        selected.append(metric_obj)
        used.append(requested)
        print(f"[RAGAS_METRIC] requested={requested} using={metric_name_found}", flush=True)

    return selected, used, skipped

def _result_to_report(result: Any, records: list[dict[str, Any]], used_metrics: list[str]):
    if hasattr(result, "to_pandas"):
        rows = result.to_pandas().to_dict(orient="records")
    elif hasattr(result, "to_dict"):
        raw = result.to_dict()
        rows = raw if isinstance(raw, list) else []
    elif isinstance(result, dict):
        rows = []
    else:
        rows = []

    summary = {}

    if isinstance(result, dict):
        for name in used_metrics:
            if isinstance(result.get(name), (int, float)):
                summary[name] = float(result[name])

    for name in used_metrics:
        values = [row.get(name) for row in rows if isinstance(row.get(name), (int, float))]
        if values:
            summary[name] = sum(float(v) for v in values) / len(values)

    per_query = []
    for idx, record in enumerate(records):
        row = rows[idx] if idx < len(rows) else {}
        scores = {}
        for name in used_metrics:
            if isinstance(row.get(name), (int, float)):
                scores[name] = float(row[name])

        per_query.append({
            "query_id": record.get("query_id", str(idx)),
            "query": record.get("question", ""),
            "scores": scores,
        })

    return summary, per_query

def _build_dataset(records: list[dict[str, Any]]):
    try:
        from datasets import Dataset
    except Exception as exc:
        print("Failed to import datasets:", repr(exc), file=sys.stderr)
        raise

    # RAGAS commonly expects these columns.
    dataset_records = []
    for r in records:
        dataset_records.append({
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r.get("ground_truth", ""),
            "reference": r.get("reference", ""),
        })

    return Dataset.from_list(dataset_records)

def run(args: argparse.Namespace) -> int:
    try:
        items = load_eval_results(args.eval_results)
    except Exception as exc:
        print(f"Failed to read eval results: {exc}", file=sys.stderr)
        return 2

    try:
        records, skipped_examples = build_ragas_records(
            items,
            max_examples=args.max_examples,
            require_contexts=args.require_contexts,
        )
    except Exception as exc:
        print(f"RAGAS evaluation requires contexts: {exc}", file=sys.stderr)
        return 2

    if not records:
        print("No examples with contexts available for RAGAS evaluation", file=sys.stderr)
        return 2

    try:
        from ragas import evaluate
    except Exception as exc:
        print("Failed to import ragas.evaluate.", file=sys.stderr)
        print("Actual exception:", repr(exc), file=sys.stderr)
        traceback.print_exc()
        return 2

    requested = _parse_metrics(args.metrics)
    has_gold = any(r.get("ground_truth") for r in records)

    try:
        metric_objects, used_metrics, skipped_metrics = _load_metric_objects(requested, has_gold)
    except Exception:
        return 2

    if not metric_objects:
        print("No requested RAGAS metrics are available to run.", file=sys.stderr)
        print("Requested:", requested, file=sys.stderr)
        print("Skipped:", skipped_metrics, file=sys.stderr)
        return 2

    try:
        dataset = _build_dataset(records)
    except Exception:
        return 2

    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        judge_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=os.getenv("LLM_MODEL", "qwen3.7-plus"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                temperature=0,
                timeout=120,
                max_retries=2,
            )
        )

        from ragas.run_config import RunConfig

        run_config = RunConfig(
            timeout=300,
            max_retries=2,
            max_wait=60,
            max_workers=1,
        )

        result = evaluate(
            dataset,
            metrics=metric_objects,
            llm=judge_llm,
            run_config=run_config,
            raise_exceptions=True,
        )

        
    except Exception as exc:
        print("RAGAS evaluation failed.", file=sys.stderr)
        print("Actual exception:", repr(exc), file=sys.stderr)
        traceback.print_exc()
        return 2

    summary_scores, per_query_scores = _result_to_report(result, records, used_metrics)

    report = {
        "input": str(args.eval_results),
        "num_input_examples": len(items if args.max_examples is None else items[:args.max_examples]),
        "num_evaluated_examples": len(records),
        "num_skipped_examples": len(skipped_examples),
        "used_metrics": used_metrics,
        "skipped_metrics": skipped_metrics,
        "summary_scores": summary_scores,
        "per_query_scores": per_query_scores,
        "skipped_examples": skipped_examples,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote", out)
    print("summary_scores =", json.dumps(summary_scores, ensure_ascii=False))
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAGAS answer/context quality evaluation on saved eval results.")
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--require-contexts", action="store_true")
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--embedding-provider", default=None)
    parser.add_argument("--skip-reference-metrics-if-no-gold", action="store_true", default=True)
    return parser

def main() -> int:
    return run(build_parser().parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
