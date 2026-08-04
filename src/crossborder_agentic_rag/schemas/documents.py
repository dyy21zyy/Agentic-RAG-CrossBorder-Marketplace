"""Document schemas for normalized source records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crossborder_agentic_rag.constants import SOURCE_TYPES
from crossborder_agentic_rag.schemas._json import json_safe
from crossborder_agentic_rag.schemas.images import ImageAsset


@dataclass(slots=True)
class NormalizedDocument:
    """A normalized source document emitted by future parser stages."""

    doc_id: str
    source_type: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    images: list[ImageAsset] = field(default_factory=list)

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
        if not isinstance(self.images, list):
            raise TypeError("images must be a list")

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_type": self.source_type,
            "title": self.title,
            "content": self.content,
            "metadata": json_safe(self.metadata, "metadata"),
            "images": [image.to_dict() for image in self.images],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizedDocument":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        values = dict(data)
        values["images"] = [ImageAsset.from_dict(image) for image in values.get("images", [])]
        return cls(**values)
