"""Local trace resources for MCP clients."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TRACE_ID_RE = re.compile(r"^trace-[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$|^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def get_trace_resource(trace_id: str, trace_dir: Path) -> dict[str, Any]:
    if not TRACE_ID_RE.fullmatch(trace_id):
        return {"trace_id": trace_id, "error": "INVALID_TRACE_ID"}
    base = Path(trace_dir).resolve()
    path = (base / f"{trace_id}.jsonl").resolve()
    try:
        path.relative_to(base)
    except ValueError:
        return {"trace_id": trace_id, "error": "INVALID_TRACE_ID"}
    if not path.is_file():
        return {"trace_id": trace_id, "error": "TRACE_NOT_FOUND"}

    with path.open(encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    return {"trace_id": trace_id, "events": events}
