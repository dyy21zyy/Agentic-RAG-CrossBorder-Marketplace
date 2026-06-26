"""Result and agent state schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
from crossborder_agentic_rag.schemas.queries import QueryPlan


@dataclass(slots=True)
class AgentState:
    """Mutable state container for future Agentic RAG workflow stages."""

    query: str
    normalized_query: str = ""
    query_type: str = ""
    expected_answer_type: str = ""
    retrieval_route: str = ""
    retrieval_plan: list[QueryPlan] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    sql_results: list[Any] = field(default_factory=list)
    retrieved_evidence: list[EvidenceChunk] = field(default_factory=list)
    reranked_evidence: list[EvidenceChunk] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    iterations: int = 0
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not self.query.strip():
            raise ValueError("query must be non-empty")
        if not isinstance(self.iterations, int):
            raise TypeError("iterations must be an integer")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")

    def add_trace(self, step: str) -> None:
        if not isinstance(step, str):
            raise TypeError("step must be a string")
        self.trace.append(step)

    def add_tool_call(self, tool: str, payload: dict[str, Any] | None = None) -> None:
        if not isinstance(tool, str):
            raise TypeError("tool must be a string")
        if payload is not None and not isinstance(payload, dict):
            raise TypeError("payload must be a dict or None")
        self.tool_calls.append({"tool": tool, "payload": {} if payload is None else payload})
