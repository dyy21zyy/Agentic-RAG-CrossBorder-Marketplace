"""Run a stable single-turn IP risk screening query."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crossborder_agentic_rag.agentic.dispatcher import ToolDispatcher
from crossborder_agentic_rag.agentic.runtime import RiskScreeningRuntime


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Run a preliminary IP risk screening query")
    parser.add_argument("query")
    parser.add_argument("--target-market", action="append", default=["US"])
    parser.add_argument("--scope", action="append", default=["trademark", "patent", "litigation"])
    parser.add_argument("--output-json", action="store_true")
    args = parser.parse_args(actual_argv)
    if "--target-market" in actual_argv:
        args.target_market = args.target_market[1:]
    if "--scope" in actual_argv:
        args.scope = args.scope[3:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = RiskScreeningRuntime(dispatcher=ToolDispatcher(), llm=None)
    report = runtime.run(args.query, target_markets=args.target_market, scope=args.scope)
    if args.output_json:
        payload = json.dumps(report.to_dict(), ensure_ascii=False)
        print(payload.encode("ascii", "backslashreplace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
