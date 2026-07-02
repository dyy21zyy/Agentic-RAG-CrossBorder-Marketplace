"""Legacy policy parser placeholder.

Policy documents are intentionally unsupported in the current trademark,
patent, and litigation pipeline.
"""
from __future__ import annotations
from pathlib import Path
from crossborder_agentic_rag.ingestion.io_utils import ensure_input_dir
from crossborder_agentic_rag.ingestion.io_utils import init_report


def parse_policy_directory(input_dir: str | Path):
    base = ensure_input_dir(input_dir)
    report = init_report("unsupported_legacy_policy", base)
    report["warnings"].append("policy source_type is not supported in the current pipeline")
    return [], report
