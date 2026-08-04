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


class _StubLlm:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def complete_structured(self, messages, schema_name):
        if self.error:
            raise self.error
        return self.result


def test_plan_tools_filters_invalid_and_out_of_scope_llm_steps():
    llm = _StubLlm(
        {
            "tool_plan": [
                {
                    "tool": "trademark_search_tool",
                    "query": "custom trademark query",
                    "retrieval_mode": "hybrid_rerank",
                    "required_evidence": "trademark",
                },
                {
                    "tool": "patent_search_tool",
                    "query": "out of scope",
                    "retrieval_mode": "hybrid_rerank",
                    "required_evidence": "patent",
                },
                {
                    "tool": "unsupported_tool",
                    "query": "unsupported",
                    "retrieval_mode": "hybrid_rerank",
                    "required_evidence": "trademark",
                },
                {
                    "tool": "trademark_search_tool",
                    "query": "wrong mode",
                    "retrieval_mode": "dense",
                    "required_evidence": "trademark",
                },
                {"tool": "trademark_search_tool", "retrieval_mode": "hybrid_rerank"},
            ]
        }
    )

    plan = plan_tools("smart phone case", ["trademark"], llm=llm)

    assert plan == [
        {
            "tool": "trademark_search_tool",
            "query": "custom trademark query",
            "retrieval_mode": "hybrid_rerank",
            "required_evidence": "trademark",
        }
    ]


def test_plan_tools_falls_back_when_llm_raises():
    llm = _StubLlm(error=RuntimeError("provider unavailable"))

    plan = plan_tools("smart phone case", ["patent"], llm=llm)

    assert plan == [
        {
            "tool": "patent_search_tool",
            "query": "technical features patent claims smart phone case",
            "retrieval_mode": "hybrid_rerank",
            "required_evidence": "patent",
        }
    ]
