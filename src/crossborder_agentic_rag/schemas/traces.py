"""Trace event schema without model reasoning content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crossborder_agentic_rag.schemas._json import json_safe


@dataclass(slots=True)
class TraceEvent:
    trace_id: str
    step: str
    event_type: str
    payload: dict[str, Any]
    timestamp: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, dict):
            raise TypeError("payload must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "step": self.step,
            "event_type": self.event_type,
            "payload": json_safe(self.payload, "payload"),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceEvent":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        return cls(**data)
