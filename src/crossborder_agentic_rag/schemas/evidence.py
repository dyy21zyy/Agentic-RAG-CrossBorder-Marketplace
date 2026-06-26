"""Evidence chunk schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crossborder_agentic_rag.constants import SOURCE_TYPES


@dataclass(slots=True)
class EvidenceChunk:
    """A retrieval-ready evidence chunk with source metadata."""

    chunk_id: str
    doc_id: str
    source_type: str
    source_subtype: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str):
            raise TypeError("chunk_id must be a string")
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must be non-empty")
        if not isinstance(self.doc_id, str):
            raise TypeError("doc_id must be a string")
        if not self.doc_id.strip():
            raise ValueError("doc_id must be non-empty")
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        if not isinstance(self.source_subtype, str):
            raise TypeError("source_subtype must be a string")
        if not self.source_subtype.strip():
            raise ValueError("source_subtype must be non-empty")
        if not isinstance(self.title, str):
            raise TypeError("title must be a string")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        try:
            self.score = float(self.score)
        except (TypeError, ValueError) as exc:
            raise TypeError("score must be convertible to float") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_type": self.source_type,
            "source_subtype": self.source_subtype,
            "title": self.title,
            "content": self.content,
            "metadata": dict(self.metadata),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceChunk":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        return cls(**data)
