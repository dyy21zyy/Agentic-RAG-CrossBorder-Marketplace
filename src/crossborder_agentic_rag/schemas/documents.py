"""Document schemas for normalized source records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crossborder_agentic_rag.constants import SOURCE_TYPES


@dataclass(slots=True)
class NormalizedDocument:
    """A normalized source document emitted by future parser stages."""

    doc_id: str
    source_type: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str):
            raise TypeError("doc_id must be a string")
        if not self.doc_id.strip():
            raise ValueError("doc_id must be non-empty")
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        if not isinstance(self.title, str):
            raise TypeError("title must be a string")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_type": self.source_type,
            "title": self.title,
            "content": self.content,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizedDocument":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        return cls(**data)
