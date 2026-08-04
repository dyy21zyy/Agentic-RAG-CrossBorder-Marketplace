from crossborder_agentic_rag.agentic.normalizer import normalize_user_query
from crossborder_agentic_rag.agentic.planner import plan_tools
from crossborder_agentic_rag.agentic.rewriter import rewrite_for_scenario


def test_normalize_user_query_defaults_market_to_us():
    normalized = normalize_user_query("Can I sell a smart phone case?", None)
    assert normalized["query"] == "Can I sell a smart phone case?"
    assert normalized["target_markets"] == ["US"]


def test_rewrite_for_scenario_builds_domain_queries():
    rewritten = rewrite_for_scenario("smart phone case", ["trademark", "patent", "litigation"])
    assert "brand logo goods services smart phone case" in rewritten["trademark"]
    assert "technical features patent claims smart phone case" in rewritten["patent"]
    assert "litigation case asserted patent smart phone case" in rewritten["litigation"]


def test_plan_tools_uses_scope_without_llm():
    plan = plan_tools("smart phone case", ["trademark", "patent"], llm=None)
    assert [step["tool"] for step in plan] == ["trademark_search_tool", "patent_search_tool"]
    assert plan[0]["retrieval_mode"] == "hybrid_rerank"
