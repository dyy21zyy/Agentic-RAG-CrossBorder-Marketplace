from crossborder_agentic_rag.reports.builder import build_risk_screening_report
from crossborder_agentic_rag.reports.citation_audit import audit_report_citations
from crossborder_agentic_rag.schemas import EvidenceHit, RiskVerdict


def make_hit(source_type="trademark"):
    return EvidenceHit(
        evidence_id="E1",
        chunk_id="trademark:1:chunk:0",
        source_type=source_type,
        title="Trademark evidence",
        content="Registered mark evidence",
        citation="[trademark:1:chunk:0] Trademark evidence",
        rank=1,
        score=1.0,
        retrieval_mode="hybrid_rerank",
        tool_name="trademark_search_tool",
    )


def test_report_with_evidence_returns_caution():
    report = build_risk_screening_report(
        query="Can I sell this product in the US?",
        target_markets=["US"],
        scope=["trademark"],
        evidence_hits=[make_hit()],
        missing_evidence=[],
        trace_id="trace-1",
    )
    assert report.overall_verdict == RiskVerdict.CAUTION
    assert report.risk_cards["caution"] == 1
    assert report.evidence_items[0].evidence_id == "E1"


def test_report_without_evidence_is_insufficient():
    report = build_risk_screening_report(
        query="Can I sell this product in the US?",
        target_markets=["US"],
        scope=["trademark"],
        evidence_hits=[],
        missing_evidence=["trademark"],
        trace_id="trace-1",
    )
    assert report.overall_verdict == RiskVerdict.INSUFFICIENT_EVIDENCE


def test_report_without_hits_and_missing_evidence_describes_no_risk():
    report = build_risk_screening_report(
        query="q",
        target_markets=["US"],
        scope=["trademark"],
        evidence_hits=[],
        missing_evidence=[],
        trace_id="trace-1",
    )
    assert report.overall_verdict == RiskVerdict.NO_RISK_FOUND
    assert "证据不足" not in report.country_summaries[0]["summary"]
    assert all("命中" not in recommendation for recommendation in report.action_recommendations)


def test_citation_audit_rejects_missing_evidence_reference():
    report = build_risk_screening_report(
        query="q",
        target_markets=["US"],
        scope=["trademark"],
        evidence_hits=[make_hit()],
        missing_evidence=[],
        trace_id="trace-1",
    )
    report.module_results[0]["evidence_ids"] = ["UNKNOWN"]
    result = audit_report_citations(report)
    assert result["valid_citation_rate"] == 0.0
    assert result["citation_coverage"] == 0.0
    assert result["unsupported_claim_count"] == 1


def test_citation_audit_rejects_corrupt_evidence_hit_citation():
    report = build_risk_screening_report(
        query="q",
        target_markets=["US"],
        scope=["trademark"],
        evidence_hits=[make_hit()],
        missing_evidence=[],
        trace_id="trace-1",
    )
    report.evidence_items[0].citation = "[other-chunk] Wrong source"

    result = audit_report_citations(report)

    assert result["valid_citation_rate"] < 1.0
    assert result["unsupported_claim_count"] == 2


def test_citation_audit_rejects_country_summary_claim_without_evidence_refs():
    report = build_risk_screening_report(
        query="q",
        target_markets=["US"],
        scope=["trademark"],
        evidence_hits=[make_hit()],
        missing_evidence=[],
        trace_id="trace-1",
    )
    report.country_summaries = [
        {"country": "US", "verdict": "caution", "summary": "Risk claim present"}
    ]

    result = audit_report_citations(report)

    assert result["unsupported_claim_count"] == 1
