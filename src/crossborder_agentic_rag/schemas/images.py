"""Image enrichment contract for normalized evidence objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crossborder_agentic_rag.schemas._json import json_safe


@dataclass(slots=True)
class ImageAsset:
    image_id: str
    source_doc_id: str
    storage_path: str = ""
    caption: str = ""
    ocr_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("image_id must be non-empty")
        if not self.source_doc_id.strip():
            raise ValueError("source_doc_id must be non-empty")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "source_doc_id": self.source_doc_id,
            "storage_path": self.storage_path,
            "caption": self.caption,
            "ocr_text": self.ocr_text,
            "metadata": json_safe(self.metadata, "metadata"),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageAsset":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        return cls(**data)
