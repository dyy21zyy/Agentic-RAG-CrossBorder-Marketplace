"""JSON-friendly contracts exposed to MCP clients."""

from __future__ import annotations

from typing import Any


def query_ip_risk_tool(payload: dict[str, Any], runtime) -> dict[str, Any]:
    """Run one risk screen and adapt its report to MCP content blocks."""
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("INVALID_INPUT: query must be non-empty")

    report = runtime.run(
        query,
        target_markets=payload.get("target_markets"),
        scope=payload.get("scope"),
    )
    return {
        "structuredContent": report.to_dict(),
        "content": [
            {
                "type": "text",
                "text": f"{report.overall_verdict.value}: {len(report.evidence_items)} evidence items",
            }
        ],
    }


def search_evidence_tool(payload: dict[str, Any], dispatcher) -> dict[str, Any]:
    """Dispatch an evidence lookup and serialize returned evidence hits."""
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("INVALID_INPUT: query must be non-empty")

    action = dict(payload)
    action.setdefault("tool", "evidence_search_tool")
    hits = dispatcher.run(action)
    evidence = [hit.to_dict() if hasattr(hit, "to_dict") else dict(hit) for hit in (hits or [])]
    return {
        "structuredContent": {"evidence_items": evidence},
        "content": [{"type": "text", "text": f"{len(evidence)} evidence items"}],
    }
