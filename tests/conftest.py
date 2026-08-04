"""Shared pytest configuration for stable Windows subprocess behavior."""

from __future__ import annotations

import os


os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
