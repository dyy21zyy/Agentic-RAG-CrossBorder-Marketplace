"""Optional MCP server entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crossborder_agentic_rag.agentic.dispatcher import ToolDispatcher
from crossborder_agentic_rag.agentic.runtime_factory import build_offline_template_runtime
from crossborder_agentic_rag.mcp_server.resources import get_trace_resource
from crossborder_agentic_rag.mcp_server.tools import query_ip_risk_tool, search_evidence_tool


def create_mcp_server(runtime=None, dispatcher=None, trace_dir: str | Path = "traces") -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "MCP server requires the optional 'mcp' package; install the mcp extra to launch it."
        ) from exc

    resolved_runtime = runtime or build_offline_template_runtime()
    resolved_dispatcher = dispatcher or getattr(resolved_runtime, "dispatcher", None) or ToolDispatcher()
    server = FastMCP("crossborder-ip-risk-agentic-rag")

    @server.tool(name="query_ip_risk")
    def _query_ip_risk(payload: dict[str, Any]) -> dict[str, Any]:
        return query_ip_risk_tool(payload, runtime=resolved_runtime)

    @server.tool(name="search_evidence")
    def _search_evidence(payload: dict[str, Any]) -> dict[str, Any]:
        return search_evidence_tool(payload, resolved_dispatcher)

    @server.resource("trace://{trace_id}")
    def _trace(trace_id: str) -> dict[str, Any]:
        return get_trace_resource(trace_id, Path(trace_dir))

    return server


def main() -> None:
    try:
        server = create_mcp_server()
    except RuntimeError as exc:
        print(str(exc))
        return
    if hasattr(server, "run"):
        server.run()
    else:
        print("MCP server registered query, search, and trace endpoints.")
