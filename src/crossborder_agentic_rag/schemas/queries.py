"""Query planning schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crossborder_agentic_rag.constants import RETRIEVAL_ROUTES, SOURCE_TYPES


@dataclass(slots=True)
class QueryPlan:
    """A minimal retrieval plan produced by future planning stages."""

    query: str
    query_type: str
    expected_answer_type: str
    retrieval_route: str
    filters: dict[str, Any] = field(default_factory=dict)
    source_types: list[str] = field(default_factory=list)
    top_k: int = 20

    def __post_init__(self) -> None:
        for field_name in ("query", "query_type", "expected_answer_type"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.retrieval_route not in RETRIEVAL_ROUTES:
            raise ValueError(f"retrieval_route must be one of {sorted(RETRIEVAL_ROUTES)}")
        if not isinstance(self.filters, dict):
            raise TypeError("filters must be a dict")
        if not isinstance(self.source_types, list):
            raise TypeError("source_types must be a list")
        invalid = [source_type for source_type in self.source_types if source_type not in SOURCE_TYPES]
        if invalid:
            raise ValueError(f"source_types contains invalid values: {invalid}")
        if not isinstance(self.top_k, int):
            raise TypeError("top_k must be an integer")
        if self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "expected_answer_type": self.expected_answer_type,
            "retrieval_route": self.retrieval_route,
            "filters": dict(self.filters),
            "source_types": list(self.source_types),
            "top_k": self.top_k,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryPlan":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        return cls(**data)
