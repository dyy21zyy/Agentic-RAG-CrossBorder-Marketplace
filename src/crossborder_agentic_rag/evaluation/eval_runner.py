"""Evaluation runner for fixture and backend-generated screening reports."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from crossborder_agentic_rag.evaluation.agent_metrics import agent_metrics
from crossborder_agentic_rag.evaluation.citation_metrics import citation_metrics
from crossborder_agentic_rag.schemas import EvaluationRun


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_fixture_evaluation(eval_rows: list[dict[str, Any]], runtime) -> EvaluationRun:
    """Run rows through a runtime and aggregate deterministic report metrics."""
    verdicts: Counter[str] = Counter()
    citation_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, float | int]] = []
    runtime_failure_count = 0

    for row in eval_rows:
        try:
            report = runtime.run(
                row["query"],
                target_markets=row.get("target_markets"),
                scope=row.get("scope"),
            )
        except Exception:
            runtime_failure_count += 1
            agent_rows.append({"tool_failure_rate": 1.0, "missing_evidence_count": 0, "evidence_count": 0})
            continue
        verdicts[report.overall_verdict.value] += 1
        citation_rows.append(citation_metrics(report))
        agent_rows.append(agent_metrics(report))

    metrics = {
        "valid_citation_rate": _mean([row["valid_citation_rate"] for row in citation_rows]),
        "citation_coverage": _mean([row["citation_coverage"] for row in citation_rows]),
        "unsupported_claim_count": sum(row["unsupported_claim_count"] for row in citation_rows),
        "tool_failure_rate": _mean([float(row["tool_failure_rate"]) for row in agent_rows]),
        "runtime_failure_count": runtime_failure_count,
        "missing_evidence_count": _mean([float(row["missing_evidence_count"]) for row in agent_rows]),
        "evidence_count": _mean([float(row["evidence_count"]) for row in agent_rows]),
    }
    return EvaluationRun(
        run_id=f"eval-{uuid4().hex}",
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset="fixture",
        sample_count=len(eval_rows),
        verdict_counts=dict(verdicts),
        metrics=metrics,
    )
