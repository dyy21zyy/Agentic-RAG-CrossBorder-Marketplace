"""Deterministic citation checks for structured screening reports."""

from typing import Any

from crossborder_agentic_rag.schemas import RiskScreeningReport, RiskVerdict


def audit_report_citations(report: RiskScreeningReport) -> dict[str, Any]:
    evidence_ids = {item.evidence_id for item in report.evidence_items}
    risk_verdicts = {verdict.value for verdict in RiskVerdict}
    claims = [
        result
        for result in report.module_results
        if result.get("verdict") in risk_verdicts
    ]
    valid_claims = 0
    referenced_ids: list[str] = []
    for claim in claims:
        claim_ids = claim.get("evidence_ids", [])
        if isinstance(claim_ids, list):
            referenced_ids.extend(claim_ids)
        if claim_ids and all(evidence_id in evidence_ids for evidence_id in claim_ids):
            valid_claims += 1

    valid_references = sum(evidence_id in evidence_ids for evidence_id in referenced_ids)
    valid_citation_rate = (
        valid_references / len(referenced_ids) if referenced_ids else 1.0
    )
    citation_coverage = (
        len(set(referenced_ids) & evidence_ids) / len(evidence_ids) if evidence_ids else 1.0
    )
    return {
        "valid_citation_rate": valid_citation_rate,
        "citation_coverage": citation_coverage,
        "unsupported_claim_count": len(claims) - valid_claims,
    }
