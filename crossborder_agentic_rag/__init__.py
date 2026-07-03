"""Repository-root import shim for the src-layout package."""
from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_SRC_PKG = _SRC / "crossborder_agentic_rag"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if _SRC_PKG.exists():
    __path__.append(str(_SRC_PKG))  # type: ignore[name-defined]
