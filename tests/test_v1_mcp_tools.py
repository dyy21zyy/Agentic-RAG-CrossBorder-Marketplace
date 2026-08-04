from crossborder_agentic_rag.mcp_server.tools import query_ip_risk_tool
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
            limitations=["本报告仅用于知识产权风险初筛和证据发现，不构成法律意见。"],
        )


def test_query_ip_risk_tool_returns_structured_content():
    response = query_ip_risk_tool(
        {"query": "Can I sell this?", "target_markets": ["US"], "scope": ["trademark"]},
        runtime=FakeRuntime(),
    )
    assert response["structuredContent"]["overall_verdict"] == "insufficient_evidence"
    assert response["content"][0]["type"] == "text"
