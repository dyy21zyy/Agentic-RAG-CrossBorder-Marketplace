"""Adapt planned tool actions to the configured retrieval backends."""

from __future__ import annotations

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

    def run(self, action: dict[str, Any]) -> list[EvidenceHit]:
        if self.retriever is None:
            return []

        tool = str(action.get("tool", ""))
        source_type = action.get("required_evidence") or TOOL_SOURCE_TYPES.get(tool)
        chunks = self.retriever.retrieve(
            str(action.get("query", "")),
            mode=action.get("retrieval_mode", "hybrid_rerank"),
            source_types=[source_type] if source_type else None,
        )
        return [
            self._to_evidence_hit(chunk, rank=index, action=action, source_type=source_type)
            for index, chunk in enumerate(chunks, start=1)
        ]

    @staticmethod
    def _to_evidence_hit(chunk, rank: int, action: dict[str, Any], source_type: str | None) -> EvidenceHit:
        chunk_id = str(_get(chunk, "chunk_id", f"chunk:{rank}"))
        title = str(_get(chunk, "title", ""))
        return EvidenceHit(
            evidence_id=f"E{rank}",
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
