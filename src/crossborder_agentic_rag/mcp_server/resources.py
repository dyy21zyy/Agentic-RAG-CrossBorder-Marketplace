"""Local trace resources for MCP clients."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_trace_resource(trace_id: str, trace_dir: Path) -> dict[str, Any]:
    path = Path(trace_dir) / f"{trace_id}.jsonl"
    if not path.is_file():
        return {"trace_id": trace_id, "error": "TRACE_NOT_FOUND"}

    with path.open(encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    return {"trace_id": trace_id, "events": events}
