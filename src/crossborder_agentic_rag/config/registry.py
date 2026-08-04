from __future__ import annotations

from collections.abc import Callable
from typing import Any


class PluginRegistry:
    def __init__(self) -> None:
        self._builders: dict[tuple[str, str], Callable[[dict[str, Any]], Any]] = {}

    def register(self, category: str, name: str, builder: Callable[[dict[str, Any]], Any]) -> None:
        key = (category.strip(), name.strip())
        if not key[0] or not key[1]:
            raise ValueError("category and name must be non-empty")
        self._builders[key] = builder

    def build(self, category: str, config: dict[str, Any]) -> Any:
        name = str(config.get("provider") or config.get("name") or "").strip()
        key = (category.strip(), name)
        if key not in self._builders:
            raise KeyError(f"No plugin registered for {category}:{name}")
        return self._builders[key](config)
