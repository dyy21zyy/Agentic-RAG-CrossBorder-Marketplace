"""Evaluation run artifact schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvaluationRun:
    run_id: str
    created_at: str
    dataset: str
    sample_count: int
    verdict_counts: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "dataset": self.dataset,
            "sample_count": self.sample_count,
            "verdict_counts": dict(self.verdict_counts),
            "metrics": dict(self.metrics),
            "artifact_paths": dict(self.artifact_paths),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationRun":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        return cls(**data)
