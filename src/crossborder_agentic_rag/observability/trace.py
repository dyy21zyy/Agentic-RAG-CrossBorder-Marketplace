"""Interfaces shared by trace backends."""

from __future__ import annotations

from typing import Protocol

from crossborder_agentic_rag.schemas import TraceEvent


class TraceSink(Protocol):
    def record(self, event: TraceEvent) -> None:
        """Persist one trace event."""


class NullTraceSink:
    def record(self, event: TraceEvent) -> None:
        del event
