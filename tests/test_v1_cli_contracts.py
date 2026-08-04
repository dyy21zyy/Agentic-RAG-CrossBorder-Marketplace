import json
import importlib.util
import subprocess
import sys
from pathlib import Path

QUERY_SCRIPT = Path(__file__).parents[1] / "scripts" / "query.py"
QUERY_MODULE_SPEC = importlib.util.spec_from_file_location("query_cli", QUERY_SCRIPT)
QUERY_MODULE = importlib.util.module_from_spec(QUERY_MODULE_SPEC)
assert QUERY_MODULE_SPEC.loader is not None
QUERY_MODULE_SPEC.loader.exec_module(QUERY_MODULE)
parse_args = QUERY_MODULE.parse_args


def test_query_cli_outputs_report_json():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/query.py",
            "Can I sell this phone case?",
            "--target-market",
            "US",
            "--scope",
            "trademark",
            "--output-json",
        ],
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["target_markets"] == ["US"]
    assert data["overall_verdict"] in {"no_risk_found", "caution", "not_recommended", "insufficient_evidence"}


def test_query_cli_uses_defaults_when_options_are_omitted():
    args = parse_args(["query"])
    assert args.target_market == ["US"]
    assert args.scope == ["trademark", "patent", "litigation"]


def test_query_cli_preserves_repeated_append_options():
    args = parse_args(
        [
            "query",
            "--target-market",
            "US",
            "--target-market",
            "CN",
            "--scope",
            "trademark",
            "--scope",
            "patent",
        ]
    )
    assert args.target_market == ["US", "CN"]
    assert args.scope == ["trademark", "patent"]


def test_query_cli_equals_form_does_not_retain_defaults():
    args = parse_args(["query", "--target-market=CN", "--scope=patent"])
    assert args.target_market == ["CN"]
    assert args.scope == ["patent"]


def test_query_cli_json_output_emits_unicode_without_ascii_escaping():
    proc = subprocess.run(
        [sys.executable, "scripts/query.py", "风险筛查", "--output-json"],
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    json.loads(proc.stdout)
    assert "\\u" not in proc.stdout


def test_query_cli_default_runtime_loads_app_config(monkeypatch):
    calls = {}

    class FakeRuntime:
        def run(self, query, target_markets=None, scope=None):
            return type(
                "Report",
                (),
                {
                    "to_dict": lambda self: {
                        "query": query,
                        "target_markets": target_markets,
                        "scope": scope,
                    }
                },
            )()

    def fake_factory(config_path):
        calls["config_path"] = config_path
        return FakeRuntime()

    monkeypatch.setattr(QUERY_MODULE, "build_runtime_from_config", fake_factory)
    monkeypatch.setattr(QUERY_MODULE, "print", lambda text: calls.setdefault("stdout", text))

    assert QUERY_MODULE.main(["query", "--output-json"]) == 0
    assert calls["config_path"].name == "app.yaml"
    assert json.loads(calls["stdout"])["target_markets"] == ["US"]


def test_query_cli_requires_explicit_offline_template_for_empty_runtime(monkeypatch):
    calls = {}

    class FakeRuntime:
        def run(self, query, target_markets=None, scope=None):
            return type("Report", (), {"to_dict": lambda self: {"offline": True}})()

    monkeypatch.setattr(QUERY_MODULE, "build_offline_template_runtime", lambda: calls.setdefault("offline", FakeRuntime()))
    monkeypatch.setattr(QUERY_MODULE, "print", lambda text: calls.setdefault("stdout", text))

    assert QUERY_MODULE.main(["query", "--offline-template", "--output-json"]) == 0
    assert "offline" in calls
    assert json.loads(calls["stdout"]) == {"offline": True}
