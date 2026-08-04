"""Evidence chunk schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crossborder_agentic_rag.constants import SOURCE_TYPES
from crossborder_agentic_rag.schemas._json import json_safe
from crossborder_agentic_rag.schemas.images import ImageAsset


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
    images: list[ImageAsset] = field(default_factory=list)

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
        if not isinstance(self.images, list):
            raise TypeError("images must be a list")
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
            "metadata": json_safe(self.metadata, "metadata"),
            "score": self.score,
            "images": [image.to_dict() for image in self.images],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceChunk":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        values = dict(data)
        values["images"] = [ImageAsset.from_dict(image) for image in values.get("images", [])]
        return cls(**values)


@dataclass(slots=True)
class EvidenceHit:
    evidence_id: str
    chunk_id: str
    source_type: str
    title: str
    content: str
    citation: str
    rank: int
    score: float
    retrieval_mode: str
    tool_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
            "title": self.title,
            "content": self.content,
            "citation": self.citation,
            "rank": self.rank,
            "score": self.score,
            "retrieval_mode": self.retrieval_mode,
            "tool_name": self.tool_name,
            "metadata": json_safe(self.metadata, "metadata"),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceHit":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        return cls(**data)
