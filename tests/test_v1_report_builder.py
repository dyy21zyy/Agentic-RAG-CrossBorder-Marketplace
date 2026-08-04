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


def test_citation_audit_rejects_missing_evidence_reference():
    report = build_risk_screening_report(
        query="q",
        target_markets=["US"],
        scope=["trademark"],
        evidence_hits=[make_hit()],
        missing_evidence=[],
        trace_id="trace-1",
    )
    result = audit_report_citations(report)
    assert result["valid_citation_rate"] == 1.0
    assert result["unsupported_claim_count"] == 0
