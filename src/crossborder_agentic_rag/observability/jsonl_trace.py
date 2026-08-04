"""Local append-only JSONL trace sink."""

from __future__ import annotations

import json
from pathlib import Path

from crossborder_agentic_rag.schemas import TraceEvent


class LocalJsonlTraceSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(self, event: TraceEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
