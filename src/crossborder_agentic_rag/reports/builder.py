"""Deterministic construction of preliminary IP risk screening reports."""

from datetime import datetime, timezone
from uuid import uuid4

from crossborder_agentic_rag.schemas import EvidenceHit, RiskScreeningReport, RiskVerdict


def build_risk_screening_report(
    query,
    target_markets,
    scope,
    evidence_hits,
    missing_evidence,
    trace_id,
):
    verdict = RiskVerdict.INSUFFICIENT_EVIDENCE if missing_evidence and not evidence_hits else RiskVerdict.CAUTION
    if evidence_hits and any(hit.source_type == "litigation" for hit in evidence_hits):
        verdict = RiskVerdict.NOT_RECOMMENDED
    if not evidence_hits and not missing_evidence:
        verdict = RiskVerdict.NO_RISK_FOUND
    risk_cards = {
        "no_risk_found": 1 if verdict == RiskVerdict.NO_RISK_FOUND else 0,
        "caution": 1 if verdict == RiskVerdict.CAUTION else 0,
        "not_recommended": 1 if verdict == RiskVerdict.NOT_RECOMMENDED else 0,
        "insufficient_evidence": 1 if verdict == RiskVerdict.INSUFFICIENT_EVIDENCE else 0,
    }
    summary_text = (
        "命中知识产权风险信号，建议人工复核后再决定是否上架。"
        if evidence_hits
        else "证据不足，需补充索引或数据后再判断。"
    )
    return RiskScreeningReport(
        report_id=f"report-{uuid4().hex}",
        trace_id=trace_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        product_profile={"query": query},
        target_markets=list(target_markets),
        screening_scope=list(scope),
        overall_verdict=verdict,
        country_summaries=[
            {"country": market, "verdict": verdict.value, "summary": summary_text}
            for market in target_markets
        ],
        risk_cards=risk_cards,
        module_results=[
            {
                "module": source,
                "verdict": verdict.value,
                "evidence_ids": [hit.evidence_id for hit in evidence_hits if hit.source_type == source],
            }
            for source in scope
        ],
        evidence_items=list(evidence_hits),
        action_recommendations=["建议人工复核命中的商标、专利或诉讼证据后再决定是否上架。"],
        missing_evidence=list(missing_evidence),
        limitations=["本报告仅用于知识产权风险初筛和证据发现，不构成法律意见。"],
    )
