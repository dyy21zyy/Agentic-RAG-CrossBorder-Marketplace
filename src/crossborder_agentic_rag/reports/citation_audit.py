"""Deterministic citation checks for structured screening reports."""

from typing import Any

from crossborder_agentic_rag.schemas import RiskScreeningReport, RiskVerdict


def _has_valid_citation_token(item) -> bool:
    return str(item.citation).strip().startswith(f"[{item.chunk_id}]")


def audit_report_citations(report: RiskScreeningReport) -> dict[str, Any]:
    evidence_ids = {item.evidence_id for item in report.evidence_items}
    evidence_by_id = {item.evidence_id: item for item in report.evidence_items}
    risk_verdicts = {verdict.value for verdict in RiskVerdict}
    metric_claims = [
        result
        for result in report.module_results
        if result.get("verdict") in risk_verdicts
    ]
    support_claims = list(metric_claims)
    support_claims.extend(
        summary
        for summary in report.country_summaries
        if summary.get("verdict") in risk_verdicts and summary.get("summary")
    )
    valid_claims = 0
    referenced_ids: list[str] = []
    for claim in support_claims:
        claim_ids = claim.get("evidence_ids", [])
        if claim_ids and all(
            evidence_id in evidence_ids
            and _has_valid_citation_token(evidence_by_id[evidence_id])
            for evidence_id in claim_ids
        ):
            valid_claims += 1
    for claim in metric_claims:
        claim_ids = claim.get("evidence_ids", [])
        if isinstance(claim_ids, list):
            referenced_ids.extend(claim_ids)

    valid_references = sum(
        evidence_id in evidence_ids
        and _has_valid_citation_token(evidence_by_id[evidence_id])
        for evidence_id in referenced_ids
    )
    valid_citation_rate = (
        valid_references / len(referenced_ids) if referenced_ids else 1.0
    )
    citation_coverage = (
        len(set(referenced_ids) & evidence_ids) / len(evidence_ids) if evidence_ids else 1.0
    )
    return {
        "valid_citation_rate": valid_citation_rate,
        "citation_coverage": citation_coverage,
        "unsupported_claim_count": len(support_claims) - valid_claims,
    }
