"""Validation and normalization for JSON-compatible schema values."""

from __future__ import annotations

import math
from typing import Any


def json_safe(value: Any, path: str = "value") -> Any:
    """Return a recursively copied JSON-compatible value or reject it."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} must contain only JSON-serializable values")
        return value
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} must contain only JSON-serializable values")
            normalized[key] = json_safe(item, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [json_safe(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} must contain only JSON-serializable values")
