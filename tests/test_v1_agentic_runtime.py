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


def test_plan_tools_falls_back_for_malformed_llm_output():
    llm = _StubLlm({"tool_plan": "not-a-list"})

    plan = plan_tools("smart phone case", ["litigation"], llm=llm)

    assert plan == [
        {
            "tool": "litigation_search_tool",
            "query": "litigation case asserted patent smart phone case",
            "retrieval_mode": "hybrid_rerank",
            "required_evidence": "litigation",
        }
    ]


from crossborder_agentic_rag.agentic.runtime import RiskScreeningRuntime
from crossborder_agentic_rag.agentic.dispatcher import ToolDispatcher
from crossborder_agentic_rag.retrieval.source_balanced import SourceBalancedRetriever
from crossborder_agentic_rag.schemas import EvidenceChunk, EvidenceHit, RiskVerdict


class FakeDispatcher:
    def run(self, action):
        return [
            EvidenceHit(
                evidence_id="E1",
                chunk_id="trademark:1:chunk:0",
                source_type="trademark",
                title="Trademark evidence",
                content="Registered mark evidence",
                citation="[trademark:1:chunk:0] Trademark evidence",
                rank=1,
                score=1.0,
                retrieval_mode=action["retrieval_mode"],
                tool_name=action["tool"],
            )
        ]


def test_runtime_returns_structured_report():
    runtime = RiskScreeningRuntime(dispatcher=FakeDispatcher(), llm=None)
    report = runtime.run(
        query="Can I sell a smart phone case?",
        target_markets=["US"],
        scope=["trademark"],
    )
    assert report.overall_verdict == RiskVerdict.CAUTION
    assert report.evidence_items[0].tool_name == "trademark_search_tool"


class RecordingTraceSink:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event.to_dict())


def test_runtime_uses_unique_trace_ids_and_structured_non_reasoning_events():
    sink = RecordingTraceSink()
    runtime = RiskScreeningRuntime(dispatcher=FakeDispatcher(), llm=None, trace_sink=sink)

    first = runtime.run("Can I sell a phone case?", target_markets=["US"], scope=["trademark"])
    second = runtime.run("Can I sell a phone case?", target_markets=["US"], scope=["trademark"])

    assert first.trace_id != second.trace_id
    assert first.trace_id.startswith("trace-")
    first_steps = [event["step"] for event in sink.events if event["trace_id"] == first.trace_id]
    assert first_steps == [
        "normalize_query",
        "query_rewrite",
        "plan_tools",
        "tool_call",
        "retrieval_result",
        "evidence_gap",
        "report",
    ]
    assert "private reasoning" not in str(sink.events)
    assert "<think>" not in str(sink.events)


def test_dispatcher_without_backends_returns_empty_hits():
    dispatcher = ToolDispatcher()

    assert dispatcher.run({"tool": "trademark_search_tool"}) == []


class SourceBalancedRetrieverLike:
    def retrieve(self, query, source_types=None):
        return [
            EvidenceChunk(
                chunk_id="trademark:1:chunk:0",
                doc_id="trademark:1",
                source_type="trademark",
                source_subtype="registration",
                title="Trademark evidence",
                content=query,
                score=0.75,
            )
        ]


def test_dispatcher_adapts_retriever_without_mode_parameter():
    dispatcher = ToolDispatcher(retriever=SourceBalancedRetrieverLike())

    hits = dispatcher.run(
        {
            "tool": "trademark_search_tool",
            "query": "smart phone case",
            "retrieval_mode": "hybrid_rerank",
            "required_evidence": "trademark",
        }
    )

    assert len(hits) == 1
    assert hits[0].retrieval_mode == "hybrid_rerank"
    assert hits[0].source_type == "trademark"


class SourceBalancedBaseRetriever:
    def retrieve(self, **kwargs):
        source_type = kwargs["source_types"][0]
        return [
            EvidenceChunk(
                chunk_id=f"{source_type}:1:chunk:0",
                doc_id=f"{source_type}:1",
                source_type=source_type,
                source_subtype="registration",
                title="Source-balanced evidence",
                content=kwargs["query"],
                score=0.9,
            )
        ]


def test_dispatcher_adapts_actual_source_balanced_retriever():
    retriever = SourceBalancedRetriever(
        SourceBalancedBaseRetriever(), per_source_k=1, final_k=1
    )
    dispatcher = ToolDispatcher(retriever=retriever)

    hits = dispatcher.run(
        {
            "tool": "trademark_search_tool",
            "query": "smart phone case",
            "retrieval_mode": "hybrid_rerank",
            "required_evidence": "trademark",
        }
    )

    assert len(hits) == 1
    assert isinstance(hits[0], EvidenceHit)
    assert hits[0].source_type == "trademark"
    assert hits[0].retrieval_mode == "hybrid_rerank"
    assert hits[0].metadata["source_balanced"] is True


def test_dispatcher_assigns_unique_evidence_ids_across_actions():
    dispatcher = ToolDispatcher(retriever=SourceBalancedRetrieverLike())
    actions = [
        {
            "tool": "trademark_search_tool",
            "query": "smart phone case",
            "retrieval_mode": "hybrid_rerank",
            "required_evidence": "trademark",
        },
        {
            "tool": "patent_search_tool",
            "query": "smart phone case",
            "retrieval_mode": "hybrid_rerank",
            "required_evidence": "patent",
        },
    ]

    hits = [hit for action in actions for hit in dispatcher.run(action)]

    assert len({hit.evidence_id for hit in hits}) == len(hits)
