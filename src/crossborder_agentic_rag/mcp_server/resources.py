"""Local trace resources for MCP clients."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TRACE_ID_RE = re.compile(r"^trace-[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$|^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _filter_trace_events(path: Path, trace_id: str) -> dict[str, Any]:
    events = [event for event in _read_jsonl(path) if event.get("trace_id") == trace_id]
    if not events:
        return {"trace_id": trace_id, "error": "TRACE_NOT_FOUND"}
    return {"trace_id": trace_id, "events": events}


def get_trace_resource(trace_id: str, trace_dir: Path) -> dict[str, Any]:
    if not TRACE_ID_RE.fullmatch(trace_id):
        return {"trace_id": trace_id, "error": "INVALID_TRACE_ID"}
    location = Path(trace_dir)
    if location.is_file():
        base = location.parent.resolve()
        path = location.resolve()
        try:
            path.relative_to(base)
        except ValueError:
            return {"trace_id": trace_id, "error": "INVALID_TRACE_ID"}
        return _filter_trace_events(path, trace_id)
    else:
        base = location.resolve()
        path = (base / f"{trace_id}.jsonl").resolve()
    try:
        path.relative_to(base)
    except ValueError:
        return {"trace_id": trace_id, "error": "INVALID_TRACE_ID"}
    if not path.is_file():
        append_log = (base / "local.jsonl").resolve()
        try:
            append_log.relative_to(base)
        except ValueError:
            return {"trace_id": trace_id, "error": "INVALID_TRACE_ID"}
        if not append_log.is_file():
            return {"trace_id": trace_id, "error": "TRACE_NOT_FOUND"}
        return _filter_trace_events(append_log, trace_id)

    events = _read_jsonl(path)
    return {"trace_id": trace_id, "events": events}
