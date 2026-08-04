"""Pure service functions used by the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crossborder_agentic_rag.schemas import RiskScreeningReport


def summarize_report(report: RiskScreeningReport) -> dict[str, Any]:
    """Return the JSON-friendly fields needed for the dashboard summary."""
    return {
        "report_id": report.report_id,
        "trace_id": report.trace_id,
        "overall_verdict": report.overall_verdict.value,
        "cards": dict(report.risk_cards),
        "target_markets": list(report.target_markets),
        "evidence_count": len(report.evidence_items),
    }


def load_eval_summary(path: str | Path) -> dict[str, Any]:
    """Load a JSON evaluation summary from disk."""
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)
