"""Optional Langfuse trace sink with a mandatory local fallback path."""

from __future__ import annotations

from datetime import datetime, timezone

from crossborder_agentic_rag.schemas import TraceEvent

from .trace import NullTraceSink, TraceSink


class LangfuseTraceSink:
    def __init__(self, enabled: bool, fallback: TraceSink | None = None) -> None:
        self.enabled = enabled
        self.fallback = fallback or NullTraceSink()
        self._client = None

    def record(self, event: TraceEvent) -> None:
        if not self.enabled:
            self.fallback.record(event)
            return
        try:
            self._record_langfuse(event)
        except Exception as exc:
            self.fallback.record(event)
            self.fallback.record(
                TraceEvent(
                    trace_id=event.trace_id,
                    step=event.step,
                    event_type="trace_backend_error",
                    payload={
                        "backend": "langfuse",
                        "error": str(exc),
                        "original_event_type": event.event_type,
                    },
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

    def _record_langfuse(self, event: TraceEvent) -> None:
        if self._client is None:
            from langfuse import Langfuse

            self._client = Langfuse()
        trace = self._client.trace(id=event.trace_id)
        trace.event(name=event.event_type, metadata=event.to_dict()["payload"])
        flush = getattr(self._client, "flush", None)
        if callable(flush):
            flush()
