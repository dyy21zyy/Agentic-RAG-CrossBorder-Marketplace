import json
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

from crossborder_agentic_rag.mcp_server.resources import get_trace_resource
from crossborder_agentic_rag.mcp_server import server as mcp_server
from crossborder_agentic_rag.mcp_server.tools import search_evidence_tool
from crossborder_agentic_rag.mcp_server.tools import query_ip_risk_tool
from crossborder_agentic_rag.schemas import EvidenceHit, RiskScreeningReport, RiskVerdict


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


class FakeDispatcher:
    def __init__(self, hits):
        self.hits = hits
        self.action = None

    def run(self, action):
        self.action = action
        return self.hits


def test_search_evidence_tool_returns_json_friendly_evidence_hits():
    hit = EvidenceHit(
        evidence_id="E1",
        chunk_id="chunk-1",
        source_type="trademark",
        title="Brand record",
        content="Trademark evidence.",
        citation="[chunk-1] Brand record",
        rank=1,
        score=0.9,
        retrieval_mode="hybrid_rerank",
        tool_name="trademark_search_tool",
    )
    dispatcher = FakeDispatcher([hit])

    response = search_evidence_tool({"query": "brand", "source_types": ["trademark"]}, dispatcher)

    assert dispatcher.action["query"] == "brand"
    assert response["structuredContent"]["evidence_items"] == [hit.to_dict()]
    json.dumps(response)


def test_search_evidence_tool_rejects_non_json_safe_mapping():
    with pytest.raises(TypeError, match="JSON-serializable"):
        search_evidence_tool({"query": "brand"}, FakeDispatcher([{"created": datetime.now()}]))


def test_get_trace_resource_reads_jsonl_trace(tmp_path: Path):
    trace_path = tmp_path / "trace-1.jsonl"
    trace_path.write_text('{"step": "normalize_query"}\n{"step": "report"}\n', encoding="utf-8")

    response = get_trace_resource("trace-1", tmp_path)

    assert response == {
        "trace_id": "trace-1",
        "events": [{"step": "normalize_query"}, {"step": "report"}],
    }


def test_get_trace_resource_filters_append_only_trace_log(tmp_path: Path):
    trace_log = tmp_path / "local.jsonl"
    trace_log.write_text(
        '{"trace_id": "trace-1", "step": "normalize_query"}\n'
        '{"trace_id": "trace-2", "step": "normalize_query"}\n'
        '{"trace_id": "trace-1", "step": "report"}\n',
        encoding="utf-8",
    )

    response = get_trace_resource("trace-1", tmp_path)

    assert response == {
        "trace_id": "trace-1",
        "events": [
            {"trace_id": "trace-1", "step": "normalize_query"},
            {"trace_id": "trace-1", "step": "report"},
        ],
    }


def test_get_trace_resource_returns_not_found_for_missing_trace(tmp_path: Path):
    assert get_trace_resource("missing", tmp_path) == {"trace_id": "missing", "error": "TRACE_NOT_FOUND"}


def test_get_trace_resource_rejects_path_traversal(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    secret = tmp_path / "secret.jsonl"
    secret.write_text('{"step": "secret"}\n', encoding="utf-8")

    response = get_trace_resource("../secret", trace_dir)

    assert response == {"trace_id": "../secret", "error": "INVALID_TRACE_ID"}


def test_create_mcp_server_registers_tools_and_trace_resource_with_fake_mcp(tmp_path: Path, monkeypatch):
    registered = {"tools": {}, "resources": {}}

    class FakeFastMCP:
        def __init__(self, name):
            self.name = name

        def tool(self, name=None):
            def decorator(func):
                registered["tools"][name or func.__name__] = func
                return func

            return decorator

        def resource(self, uri):
            def decorator(func):
                registered["resources"][uri] = func
                return func

            return decorator

    fake_mcp = types.SimpleNamespace(server=types.SimpleNamespace(fastmcp=types.SimpleNamespace(FastMCP=FakeFastMCP)))
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_mcp.server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_mcp.server.fastmcp)

    server = mcp_server.create_mcp_server(runtime=FakeRuntime(), trace_dir=tmp_path)

    assert server.name == "crossborder-ip-risk-agentic-rag"
    assert list(registered["tools"]) == ["query_ip_risk", "search_evidence"]
    assert list(registered["resources"]) == ["trace://{trace_id}"]


def test_create_mcp_server_default_query_trace_is_readable(tmp_path: Path, monkeypatch):
    registered = {"tools": {}, "resources": {}}

    class FakeFastMCP:
        def __init__(self, name):
            self.name = name

        def tool(self, name=None):
            def decorator(func):
                registered["tools"][name or func.__name__] = func
                return func

            return decorator

        def resource(self, uri):
            def decorator(func):
                registered["resources"][uri] = func
                return func

            return decorator

    fake_mcp = types.SimpleNamespace(server=types.SimpleNamespace(fastmcp=types.SimpleNamespace(FastMCP=FakeFastMCP)))
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_mcp.server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_mcp.server.fastmcp)

    mcp_server.create_mcp_server(trace_dir=tmp_path)
    response = registered["tools"]["query_ip_risk"]({"query": "brand logo phone case", "scope": ["trademark"]})
    trace_id = response["structuredContent"]["trace_id"]

    trace = registered["resources"]["trace://{trace_id}"](trace_id)

    assert [event["event_type"] for event in trace["events"]] == [
        "normalize_query",
        "query_rewrite",
        "plan_tools",
        "tool_call",
        "retrieval_result",
        "evidence_gap",
        "report",
    ]


@pytest.mark.parametrize("tool", [query_ip_risk_tool, search_evidence_tool])
def test_mcp_tools_reject_invalid_query(tool):
    dependency = FakeRuntime() if tool is query_ip_risk_tool else FakeDispatcher([])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        tool({"query": "  "}, runtime=dependency) if tool is query_ip_risk_tool else tool({"query": ""}, dependency)


def test_mcp_launcher_prints_optional_dependency_fallback():
    script = Path(__file__).parents[1] / "scripts" / "run_mcp_server.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert "optional 'mcp' package" in result.stdout
