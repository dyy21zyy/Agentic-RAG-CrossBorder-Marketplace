#!/usr/bin/env python3
"""Run optional RAGAS answer-quality evaluation on saved eval results.

This script intentionally does not compute retrieval metrics such as Precision@k,
Recall@k, MRR, MAP, or nDCG. Those remain part of the standard evaluation
pipeline. RAGAS is used only for answer/context quality metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_METRICS = "faithfulness,answer_relevancy,context_precision,context_recall"
REFERENCE_METRICS = {"context_recall", "answer_correctness", "answer_similarity"}


def load_eval_results(path: str | Path) -> list[dict[str, Any]]:
    """Load eval result objects from JSON list/dict-with-results or JSONL."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing eval results file: {p}")
    if p.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
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
        if not all(isinstance(x, dict) for x in data):
            raise ValueError("Expected all JSON list items to be objects")
        return data
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        rows = data["results"]
        if not all(isinstance(x, dict) for x in rows):
            raise ValueError('Expected all JSON "results" items to be objects')
        return rows
    raise ValueError('Expected JSON list or JSON object with a "results" list')


def _as_contexts(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(x) for x in value if x]
    return []


def build_ragas_records(
    items: list[dict[str, Any]], *, max_examples: int | None = None, require_contexts: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Convert saved EvalResult objects to RAGAS Dataset records."""
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    source_items = items[:max_examples] if max_examples is not None else items
    for idx, item in enumerate(source_items):
        query_id = str(item.get("query_id") or item.get("id") or idx)
        contexts = _as_contexts(
            item.get("retrieved_contexts") or item.get("contexts") or item.get("reference_contexts") or []
        )
        if not contexts:
            msg = {"query_id": query_id, "reason": "missing contexts"}
            if require_contexts:
                raise ValueError(f"missing contexts for query_id={query_id}")
            skipped.append(msg)
            continue
        question = str(item.get("query") or item.get("question") or "")
        answer = str(item.get("answer") or "")
        ground_truth = str(item.get("gold_answer") or item.get("reference") or "")
        record = {"question": question, "answer": answer, "contexts": contexts, "ground_truth": ground_truth}
        # Preserve ids for reporting without sending extra columns to RAGAS-dependent logic assumptions.
        record["query_id"] = query_id
        records.append(record)
    return records, skipped


def _parse_metrics(metrics: str) -> list[str]:
    return [m.strip() for m in metrics.split(",") if m.strip()]


def _load_metric_objects(metric_names: list[str], has_gold: bool) -> tuple[list[Any], list[str], dict[str, str]]:
    skipped: dict[str, str] = {}
    selected: list[Any] = []
    used: list[str] = []
    try:
        import ragas.metrics as ragas_metrics  # type: ignore
    except ImportError:
        print("ragas is not installed. Install it with: pip install ragas", file=sys.stderr)
        raise

    for name in metric_names:
        if name in REFERENCE_METRICS and not has_gold:
            skipped[name] = "Skipped because no gold_answer / reference is available"
            continue
        try:
            metric = getattr(ragas_metrics, name)
        except AttributeError:
            skipped[name] = f"Skipped because metric '{name}' is not available in this ragas version"
            continue
        selected.append(metric)
        used.append(name)
    return selected, used, skipped


def _result_to_report(result: Any, records: list[dict[str, Any]], used_metrics: list[str]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if hasattr(result, "to_pandas"):
        rows = result.to_pandas().to_dict(orient="records")
    elif hasattr(result, "to_dict"):
        raw = result.to_dict()
        rows = raw if isinstance(raw, list) else []
    else:
        rows = []

    summary: dict[str, float] = {}
    if isinstance(result, dict):
        for name in used_metrics:
            if isinstance(result.get(name), (int, float)):
                summary[name] = float(result[name])
    for name in used_metrics:
        values = [row.get(name) for row in rows if isinstance(row.get(name), (int, float))]
        if values:
            summary[name] = sum(float(v) for v in values) / len(values)

    per_query: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        row = rows[idx] if idx < len(rows) else {}
        scores = {name: float(row[name]) for name in used_metrics if isinstance(row.get(name), (int, float))}
        per_query.append({"query_id": record.get("query_id", str(idx)), "query": record.get("question", ""), "scores": scores})
    return summary, per_query


def run(args: argparse.Namespace) -> int:
    try:
        items = load_eval_results(args.eval_results)
    except FileNotFoundError as exc:
        print(f"Missing eval results file: {args.eval_results}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Failed to read eval results: {exc}", file=sys.stderr)
        return 2

    try:
        records, skipped_examples = build_ragas_records(items, max_examples=args.max_examples, require_contexts=args.require_contexts)
    except ValueError as exc:
        print(f"RAGAS evaluation requires contexts: {exc}", file=sys.stderr)
        return 2
    if not records:
        print("No examples with contexts available for RAGAS evaluation", file=sys.stderr)
        return 2

    try:
        from ragas import evaluate  # type: ignore
    except ImportError:
        print("ragas is not installed. Install it with: pip install ragas", file=sys.stderr)
        return 2
    try:
        from datasets import Dataset  # type: ignore
    except ImportError:
        print("datasets is not installed. Install it with: pip install datasets", file=sys.stderr)
        return 2

    requested = _parse_metrics(args.metrics)
    has_gold = any(r.get("ground_truth") for r in records)
    try:
        metric_objects, used_metrics, skipped_metrics = _load_metric_objects(requested, has_gold)
    except ImportError:
        return 2
    if not metric_objects:
        print("No requested RAGAS metrics are available to run", file=sys.stderr)
        return 2

    dataset_records = [{k: v for k, v in r.items() if k != "query_id"} for r in records]
    errors: list[str] = []
    try:
        result = evaluate(Dataset.from_list(dataset_records), metrics=metric_objects)
    except Exception as exc:
        print(f"RAGAS evaluation failed: {exc}", file=sys.stderr)
        return 2

    summary_scores, per_query_scores = _result_to_report(result, records, used_metrics)
    report = {
        "input": str(args.eval_results),
        "num_input_examples": len(items if args.max_examples is None else items[: args.max_examples]),
        "num_evaluated_examples": len(records),
        "num_skipped_examples": len(skipped_examples),
        "used_metrics": used_metrics,
        "skipped_metrics": skipped_metrics,
        "summary_scores": summary_scores,
        "per_query_scores": per_query_scores,
        "skipped_examples": skipped_examples,
        "errors": errors,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAGAS answer/context quality evaluation on saved eval results.")
    parser.add_argument("--eval-results", required=True, help="Path to eval_results.json or results.jsonl")
    parser.add_argument("--output", required=True, help="Path to write RAGAS JSON report")
    parser.add_argument("--metrics", default=DEFAULT_METRICS, help="Comma-separated RAGAS metric names")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional maximum examples to evaluate")
    parser.add_argument("--require-contexts", action="store_true", help="Fail if any selected example lacks contexts")
    parser.add_argument("--llm-provider", default=None, help="Reserved for environment-specific RAGAS configuration")
    parser.add_argument("--embedding-provider", default=None, help="Reserved for environment-specific RAGAS configuration")
    parser.add_argument("--skip-reference-metrics-if-no-gold", action="store_true", default=True, help="Skip metrics requiring references when no gold answer exists")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
