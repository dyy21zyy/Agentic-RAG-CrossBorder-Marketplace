"""Citation metrics for structured screening reports."""

from __future__ import annotations

from typing import Any

from crossborder_agentic_rag.reports.citation_audit import audit_report_citations
from crossborder_agentic_rag.schemas import RiskScreeningReport


def citation_metrics(report: RiskScreeningReport) -> dict[str, Any]:
    """Return the deterministic citation metrics for one report."""
    return audit_report_citations(report)


compute_citation_metrics = citation_metrics
