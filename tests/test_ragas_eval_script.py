from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ragas_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_ragas_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_ragas_eval_help_returns_zero():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode == 0
    assert "--eval-results" in proc.stdout


def test_load_eval_results_json_list(tmp_path):
    module = _load_module()
    path = tmp_path / "eval_results.json"
    path.write_text(json.dumps([
        {
            "query_id": "Q001",
            "query": "What patent evidence relates to USB charging backpacks?",
            "answer": "The evidence mentions USB charging backpacks.",
            "retrieved_contexts": ["Patent evidence about USB charging backpacks."],
            "gold_answer": "",
        }
    ]), encoding="utf-8")
    rows = module.load_eval_results(path)
    records, skipped = module.build_ragas_records(rows)
    assert len(records) == 1
    assert records[0]["question"].startswith("What patent evidence")
    assert records[0]["contexts"] == ["Patent evidence about USB charging backpacks."]
    assert skipped == []


def test_load_eval_results_jsonl(tmp_path):
    module = _load_module()
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"query_id": "Q001", "query": "q", "answer": "a", "contexts": ["c"]}) + "\n", encoding="utf-8")
    rows = module.load_eval_results(path)
    records, skipped = module.build_ragas_records(rows)
    assert len(rows) == 1
    assert records[0]["answer"] == "a"
    assert skipped == []


def test_missing_contexts_are_skipped(tmp_path):
    module = _load_module()
    rows = [{"query_id": "Q002", "query": "q", "answer": "a"}]
    records, skipped = module.build_ragas_records(rows)
    assert records == []
    assert skipped == [{"query_id": "Q002", "reason": "missing contexts"}]


def test_no_contexts_cli_error_is_clear(tmp_path):
    path = tmp_path / "eval_results.json"
    path.write_text(json.dumps([{"query_id": "Q002", "query": "q", "answer": "a"}]), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--eval-results", str(path), "--output", str(tmp_path / "out.json")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 2
    assert "No examples with contexts available for RAGAS evaluation" in proc.stderr
