"""Run a stable single-turn IP risk screening query."""

from __future__ import annotations

import argparse
import builtins
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crossborder_agentic_rag.agentic.runtime_factory import (
    build_offline_template_runtime,
    build_runtime_from_config,
)

print = builtins.print


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a preliminary IP risk screening query")
    parser.add_argument("query")
    parser.add_argument("--target-market", action="append", default=None)
    parser.add_argument("--scope", action="append", default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "app.yaml")
    parser.add_argument("--offline-template", action="store_true")
    parser.add_argument("--output-json", action="store_true")
    args = parser.parse_args(argv)
    if args.target_market is None:
        args.target_market = ["US"]
    if args.scope is None:
        args.scope = ["trademark", "patent", "litigation"]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = build_offline_template_runtime() if args.offline_template else build_runtime_from_config(args.config)
    report = runtime.run(args.query, target_markets=args.target_market, scope=args.scope)
    if args.output_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
