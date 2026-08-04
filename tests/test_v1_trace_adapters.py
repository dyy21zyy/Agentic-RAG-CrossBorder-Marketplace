import json

from crossborder_agentic_rag.observability.jsonl_trace import LocalJsonlTraceSink
from crossborder_agentic_rag.observability.langfuse_trace import LangfuseTraceSink
from crossborder_agentic_rag.agentic.runtime import RiskScreeningRuntime
from crossborder_agentic_rag.schemas import EvidenceHit
from crossborder_agentic_rag.schemas import TraceEvent


def test_local_jsonl_trace_sink_writes_event(tmp_path):
    path = tmp_path / "trace.jsonl"
    sink = LocalJsonlTraceSink(path)
    sink.record(
        TraceEvent(
            trace_id="trace-1",
            step="planner",
            event_type="tool_plan",
            payload={"tool_count": 1},
            timestamp="2026-08-03T00:00:00Z",
        )
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event_type"] == "tool_plan"


class _RecordingSink:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


def test_disabled_langfuse_sink_uses_fallback():
    fallback = _RecordingSink()
    event = TraceEvent("trace-1", "planner", "tool_plan", {"tool_count": 1}, "now")

    LangfuseTraceSink(enabled=False, fallback=fallback).record(event)

    assert fallback.events == [event]


def test_langfuse_failure_preserves_event_and_records_backend_error():
    fallback = _RecordingSink()
    event = TraceEvent("trace-1", "planner", "tool_plan", {"tool_count": 1}, "now")
    sink = LangfuseTraceSink(enabled=True, fallback=fallback)
    sink._record_langfuse = lambda event: (_ for _ in ()).throw(RuntimeError("langfuse unavailable"))

    sink.record(event)

    assert fallback.events[0] == event
    assert fallback.events[1].event_type == "trace_backend_error"
    assert fallback.events[1].payload["original_event_type"] == "tool_plan"


def test_runtime_records_structural_events_without_reasoning_content():
    class Dispatcher:
        def run(self, action):
            return [
                EvidenceHit(
                    evidence_id="E1",
                    chunk_id="trademark:1:chunk:0",
                    source_type="trademark",
                    title="Trademark evidence",
                    content="evidence",
                    citation="[trademark:1:chunk:0] Trademark evidence",
                    rank=1,
                    score=1.0,
                    retrieval_mode=action["retrieval_mode"],
                    tool_name=action["tool"],
                )
            ]

    sink = _RecordingSink()
    RiskScreeningRuntime(Dispatcher(), trace_sink=sink).run(
        "Can I sell a smart phone case?", target_markets=["US"], scope=["trademark"]
    )

    assert [event.event_type for event in sink.events] == [
        "normalize_query",
        "query_rewrite",
        "plan_tools",
        "tool_call",
        "retrieval_result",
        "evidence_gap",
        "report",
    ]
    assert all("reasoning" not in event.to_dict()["payload"] for event in sink.events)
