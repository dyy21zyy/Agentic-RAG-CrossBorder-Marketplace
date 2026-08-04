"""Configuration settings loader."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AppConfig:
    llm: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)
    mcp: dict[str, Any] = field(default_factory=dict)
    dashboard: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)


def load_app_config(path: str | Path) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError("app config must be a mapping")
    return AppConfig(
        llm=dict(data.get("llm") or {}),
        retrieval=dict(data.get("retrieval") or {}),
        observability=dict(data.get("observability") or {}),
        mcp=dict(data.get("mcp") or {}),
        dashboard=dict(data.get("dashboard") or {}),
        evaluation=dict(data.get("evaluation") or {}),
    )
