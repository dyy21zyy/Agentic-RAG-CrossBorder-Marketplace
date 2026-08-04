"""Run the optional Streamlit dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
STREAMLIT_HELP = (
    "Streamlit is not installed. Install the optional dashboard dependency "
    "with `python -m pip install -e \".[dashboard]\"` to launch the UI."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the optional Streamlit IP risk dashboard"
    )
    return parser.parse_args(argv)


def _load_dashboard_main():
    from crossborder_agentic_rag.dashboard.app import main as dashboard_main

    return dashboard_main


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    try:
        dashboard_main = _load_dashboard_main()
        dashboard_main()
    except ModuleNotFoundError as exc:
        if exc.name != "streamlit":
            raise
        print(STREAMLIT_HELP)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
