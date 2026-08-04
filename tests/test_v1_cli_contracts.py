import json
import subprocess
import sys


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
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["target_markets"] == ["US"]
    assert data["overall_verdict"] in {"no_risk_found", "caution", "not_recommended", "insufficient_evidence"}
