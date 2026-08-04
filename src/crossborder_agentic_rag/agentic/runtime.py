"""Single-turn runtime for structured IP risk screening."""

from __future__ import annotations

from crossborder_agentic_rag.agentic.evidence_gap import find_missing_evidence
from crossborder_agentic_rag.agentic.normalizer import normalize_user_query
from crossborder_agentic_rag.agentic.planner import plan_tools
from crossborder_agentic_rag.reports.builder import build_risk_screening_report


class RiskScreeningRuntime:
    def __init__(self, dispatcher, llm=None) -> None:
        self.dispatcher = dispatcher
        self.llm = llm

    def run(
        self,
        query: str,
        target_markets: list[str] | None = None,
        scope: list[str] | None = None,
    ):
        selected_scope = scope or ["trademark", "patent", "litigation"]
        normalized = normalize_user_query(query, target_markets)
        actions = plan_tools(str(normalized["query"]), selected_scope, llm=self.llm)
        hits = []
        for action in actions:
            hits.extend(self.dispatcher.run(action))
        missing = find_missing_evidence(selected_scope, hits)
        return build_risk_screening_report(
            query=str(normalized["query"]),
            target_markets=list(normalized["target_markets"]),
            scope=selected_scope,
            evidence_hits=hits,
            missing_evidence=missing,
            trace_id="trace-local",
        )
