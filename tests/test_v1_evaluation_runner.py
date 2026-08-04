import importlib.util
import sys
import types
from pathlib import Path

from crossborder_agentic_rag.evaluation.eval_runner import run_fixture_evaluation
from crossborder_agentic_rag.schemas import RiskScreeningReport, RiskVerdict


def _load_evaluate_script():
    path = Path(__file__).parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("task11_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRuntime:
    def run(self, query, target_markets=None, scope=None):
        return RiskScreeningReport(
            report_id="report-1",
            trace_id="trace-1",
            created_at="2026-08-03T00:00:00Z",
            product_profile={"query": query},
            target_markets=target_markets or ["US"],
            screening_scope=scope or ["trademark"],
            overall_verdict=RiskVerdict.INSUFFICIENT_EVIDENCE,
            country_summaries=[],
            risk_cards={"no_risk_found": 0, "caution": 0, "not_recommended": 0, "insufficient_evidence": 1},
            module_results=[],
            evidence_items=[],
            action_recommendations=[],
            missing_evidence=["trademark"],
            limitations=[],
        )


def test_run_fixture_evaluation_returns_summary():
    run = run_fixture_evaluation(
        [{"id": "Q1", "query": "Can I sell this?", "target_markets": ["US"], "scope": ["trademark"]}],
        runtime=FakeRuntime(),
    )
    assert run.run_id
    assert run.summary["n"] == 1
    assert run.summary["insufficient_evidence"] == 1


class FailingRuntime(FakeRuntime):
    def __init__(self):
        self.calls = 0

    def run(self, query, target_markets=None, scope=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("backend unavailable")
        return super().run(query, target_markets=target_markets, scope=scope)


def test_run_fixture_evaluation_captures_runtime_failures_and_counts_tool_failure_rate():
    run = run_fixture_evaluation(
        [
            {"id": "Q1", "query": "first", "target_markets": ["US"], "scope": ["trademark"]},
            {"id": "Q2", "query": "second", "target_markets": ["US"], "scope": ["trademark"]},
        ],
        runtime=FailingRuntime(),
    )

    assert run.sample_count == 2
    assert run.metrics["runtime_failure_count"] == 1
    assert run.metrics["tool_failure_rate"] == 0.5


def test_cli_runtime_selection_honors_configured_factory(monkeypatch):
    evaluate = _load_evaluate_script()
    monkeypatch.delenv("CROSSBORDER_EVAL_RUNTIME_FACTORY", raising=False)
    assert isinstance(evaluate._build_runtime(), evaluate.FixtureRuntime)

    configured_runtime = object()
    factory_module = types.ModuleType("task11_runtime_factory")
    factory_module.build = lambda: configured_runtime
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)
    monkeypatch.setenv(
        "CROSSBORDER_EVAL_RUNTIME_FACTORY",
        "task11_runtime_factory:build",
    )

    assert evaluate._build_runtime() is configured_runtime
