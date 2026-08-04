import importlib.util
import json
import sys

import pytest

from crossborder_agentic_rag.dashboard.services import load_eval_summary, summarize_report
from crossborder_agentic_rag.schemas import RiskScreeningReport, RiskVerdict


def test_summarize_report_counts_cards():
    report = RiskScreeningReport(
        report_id="report-1",
        trace_id="trace-1",
        created_at="2026-08-03T00:00:00Z",
        product_profile={"query": "phone case"},
        target_markets=["US"],
        screening_scope=["trademark"],
        overall_verdict=RiskVerdict.CAUTION,
        country_summaries=[],
        risk_cards={"no_risk_found": 0, "caution": 1, "not_recommended": 0, "insufficient_evidence": 0},
        module_results=[],
        evidence_items=[],
        action_recommendations=[],
        missing_evidence=[],
        limitations=[],
    )
    summary = summarize_report(report)
    assert summary["report_id"] == "report-1"
    assert summary["trace_id"] == "trace-1"
    assert summary["overall_verdict"] == "caution"
    assert summary["cards"]["caution"] == 1
    assert summary["target_markets"] == ["US"]
    assert summary["evidence_count"] == 0


def test_load_eval_summary_reads_json_object(tmp_path):
    path = tmp_path / "eval-summary.json"
    path.write_text(json.dumps({"accuracy": 0.9}), encoding="utf-8")

    assert load_eval_summary(path) == {"accuracy": 0.9}


@pytest.mark.parametrize("value", [[], "summary", 1, None])
def test_load_eval_summary_rejects_non_object_root(tmp_path, value):
    path = tmp_path / "invalid-summary.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_eval_summary(path)


def test_dashboard_app_import_does_not_import_streamlit():
    sys.modules.pop("streamlit", None)

    import crossborder_agentic_rag.dashboard.app  # noqa: F401

    assert "streamlit" not in sys.modules


def test_dashboard_launcher_import_is_safe():
    launcher_path = "scripts/run_dashboard.py"
    spec = importlib.util.spec_from_file_location("run_dashboard", launcher_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
