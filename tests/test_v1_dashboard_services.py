from crossborder_agentic_rag.dashboard.services import summarize_report
from crossborder_agentic_rag.schemas import RiskScreeningReport, RiskVerdict


def test_summarize_report_counts_cards():
    report = RiskScreeningReport(
        report_id="report-1",
        trace_id="trace-1",
        created_at="2026-08-03T00:00:00Z",
        product_profile={"query": "phone case"},
        target_markets=["US"],
        screening_scope=["trademark"],
        overall_verdict=RiskVerdict.CAUTION,
        country_summaries=[],
        risk_cards={"no_risk_found": 0, "caution": 1, "not_recommended": 0, "insufficient_evidence": 0},
        module_results=[],
        evidence_items=[],
        action_recommendations=[],
        missing_evidence=[],
        limitations=[],
    )
    summary = summarize_report(report)
    assert summary["report_id"] == "report-1"
    assert summary["cards"]["caution"] == 1
