"""Run a fixture-safe evaluation over JSONL screening queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crossborder_agentic_rag.evaluation.agent_metrics import agent_metrics
from crossborder_agentic_rag.evaluation.citation_metrics import citation_metrics
from crossborder_agentic_rag.evaluation.eval_runner import run_fixture_evaluation
from crossborder_agentic_rag.schemas import RiskScreeningReport, RiskVerdict


class FixtureRuntime:
    """Deterministic fallback runtime used when no real backend is configured."""

    def run(self, query, target_markets=None, scope=None):
        return RiskScreeningReport(
            report_id="fixture-report",
            trace_id="fixture-trace",
            created_at="2026-08-03T00:00:00Z",
            product_profile={"query": query},
            target_markets=target_markets or ["US"],
            screening_scope=scope or ["trademark"],
            overall_verdict=RiskVerdict.INSUFFICIENT_EVIDENCE,
            country_summaries=[],
            risk_cards={verdict.value: 0 for verdict in RiskVerdict},
            module_results=[],
            evidence_items=[],
            action_recommendations=[],
            missing_evidence=list(scope or ["trademark"]),
            limitations=["fixture runtime; no retrieval backend configured"],
        )


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} must be an object")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_jsonl(args.eval_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = run_fixture_evaluation(rows, FixtureRuntime())
    run.artifact_paths.update(
        {
            "summary": "summary.json",
            "citation_audit": "citation_audit.json",
            "agent_metrics": "agent_metrics.json",
        }
    )
    summary = {**run.to_dict(), "summary": run.summary}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "citation_audit.json").write_text(
        json.dumps({key: run.metrics[key] for key in ("valid_citation_rate", "citation_coverage", "unsupported_claim_count")}, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "agent_metrics.json").write_text(
        json.dumps({key: run.metrics[key] for key in ("tool_failure_rate", "missing_evidence_count", "evidence_count")}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
