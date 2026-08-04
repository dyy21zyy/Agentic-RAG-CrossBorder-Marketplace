from crossborder_agentic_rag.evaluation.eval_runner import run_fixture_evaluation
from crossborder_agentic_rag.schemas import RiskScreeningReport, RiskVerdict


class FakeRuntime:
    def run(self, query, target_markets=None, scope=None):
        return RiskScreeningReport(
            report_id="report-1",
            trace_id="trace-1",
            created_at="2026-08-03T00:00:00Z",
            product_profile={"query": query},
            target_markets=target_markets or ["US"],
            screening_scope=scope or ["trademark"],
            overall_verdict=RiskVerdict.INSUFFICIENT_EVIDENCE,
            country_summaries=[],
            risk_cards={"no_risk_found": 0, "caution": 0, "not_recommended": 0, "insufficient_evidence": 1},
            module_results=[],
            evidence_items=[],
            action_recommendations=[],
            missing_evidence=["trademark"],
            limitations=[],
        )


def test_run_fixture_evaluation_returns_summary():
    run = run_fixture_evaluation(
        [{"id": "Q1", "query": "Can I sell this?", "target_markets": ["US"], "scope": ["trademark"]}],
        runtime=FakeRuntime(),
    )
    assert run.run_id
    assert run.summary["n"] == 1
    assert run.summary["insufficient_evidence"] == 1
