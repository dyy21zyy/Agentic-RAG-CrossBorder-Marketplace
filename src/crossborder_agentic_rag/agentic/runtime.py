"""Single-turn runtime for structured IP risk screening."""

from __future__ import annotations

from datetime import datetime, timezone

from crossborder_agentic_rag.agentic.evidence_gap import find_missing_evidence
from crossborder_agentic_rag.agentic.normalizer import normalize_user_query
from crossborder_agentic_rag.agentic.planner import plan_tools
from crossborder_agentic_rag.observability.trace import TraceSink
from crossborder_agentic_rag.reports.builder import build_risk_screening_report
from crossborder_agentic_rag.schemas import RiskScreeningReport, TraceEvent


class RiskScreeningRuntime:
    def __init__(self, dispatcher, llm=None, trace_sink: TraceSink | None = None) -> None:
        self.dispatcher = dispatcher
        self.llm = llm
        self.trace_sink = trace_sink

    def _record(self, trace_id: str, step: str, event_type: str, payload: dict) -> None:
        if self.trace_sink is not None:
            self.trace_sink.record(
                TraceEvent(
                    trace_id=trace_id,
                    step=step,
                    event_type=event_type,
                    payload=payload,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

    def run(
        self,
        query: str,
        target_markets: list[str] | None = None,
        scope: list[str] | None = None,
    ) -> RiskScreeningReport:
        selected_scope = scope or ["trademark", "patent", "litigation"]
        trace_id = "trace-local"
        normalized = normalize_user_query(query, target_markets)
        self._record(
            trace_id,
            "normalize_query",
            "normalize_query",
            {
                "target_markets": list(normalized["target_markets"]),
                "scope": list(selected_scope),
            },
        )
        actions = plan_tools(str(normalized["query"]), selected_scope, llm=self.llm)
        self._record(
            trace_id,
            "plan_tools",
            "plan_tools",
            {
                "tool_count": len(actions),
                "tools": [str(action.get("tool", "")) for action in actions],
            },
        )
        hits = []
        for action in actions:
            self._record(
                trace_id,
                "tool_call",
                "tool_call",
                {"tool": str(action.get("tool", "")), "retrieval_mode": str(action.get("retrieval_mode", ""))},
            )
            hits.extend(self.dispatcher.run(action))
        missing = find_missing_evidence(selected_scope, hits)
        self._record(
            trace_id,
            "evidence_gap",
            "evidence_gap",
            {"missing_evidence": list(missing), "evidence_count": len(hits)},
        )
        report = build_risk_screening_report(
            query=str(normalized["query"]),
            target_markets=list(normalized["target_markets"]),
            scope=selected_scope,
            evidence_hits=hits,
            missing_evidence=missing,
            trace_id=trace_id,
        )
        self._record(
            trace_id,
            "report",
            "report",
            {"overall_verdict": report.overall_verdict.value, "evidence_count": len(report.evidence_items)},
        )
        return report
