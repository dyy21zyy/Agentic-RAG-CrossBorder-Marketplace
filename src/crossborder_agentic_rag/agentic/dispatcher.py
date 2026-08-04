"""Adapt planned tool actions to the configured retrieval backends."""

from __future__ import annotations

import inspect
from typing import Any

from crossborder_agentic_rag.schemas import EvidenceHit


TOOL_SOURCE_TYPES = {
    "trademark_search_tool": "trademark",
    "patent_search_tool": "patent",
    "litigation_search_tool": "litigation",
}


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class ToolDispatcher:
    def __init__(self, retriever=None, duckdb_store=None, graph_retriever=None) -> None:
        self.retriever = retriever
        self.duckdb_store = duckdb_store
        self.graph_retriever = graph_retriever
        self._evidence_count = 0

    def run(self, action: dict[str, Any]) -> list[EvidenceHit]:
        if self.retriever is None:
            return []

        tool = str(action.get("tool", ""))
        source_type = action.get("required_evidence") or TOOL_SOURCE_TYPES.get(tool)
        query = str(action.get("query", ""))
        retrieval_kwargs = {"source_types": [source_type] if source_type else None}
        try:
            parameters = inspect.signature(self.retriever.retrieve).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "mode" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            retrieval_kwargs["mode"] = action.get("retrieval_mode", "hybrid_rerank")
        chunks = self.retriever.retrieve(query, **retrieval_kwargs)
        return [
            self._to_evidence_hit(chunk, rank=index, action=action, source_type=source_type)
            for index, chunk in enumerate(chunks, start=1)
        ]

    def _next_evidence_id(self) -> int:
        self._evidence_count += 1
        return self._evidence_count

    def _to_evidence_hit(self, chunk, rank: int, action: dict[str, Any], source_type: str | None) -> EvidenceHit:
        chunk_id = str(_get(chunk, "chunk_id", f"chunk:{rank}"))
        title = str(_get(chunk, "title", ""))
        return EvidenceHit(
            evidence_id=f"E{self._next_evidence_id()}",
            chunk_id=chunk_id,
            source_type=str(_get(chunk, "source_type", source_type or "")),
            title=title,
            content=str(_get(chunk, "content", "")),
            citation=f"[{chunk_id}] {title}".strip(),
            rank=rank,
            score=float(_get(chunk, "score", 0.0) or 0.0),
            retrieval_mode=str(action.get("retrieval_mode", "hybrid_rerank")),
            tool_name=str(action.get("tool", "")),
            metadata=_get(chunk, "metadata", {}) or {},
        )
