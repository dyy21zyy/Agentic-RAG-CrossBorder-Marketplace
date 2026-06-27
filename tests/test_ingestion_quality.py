from __future__ import annotations
import importlib.util, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_ingestion_quality", ROOT / "scripts" / "check_ingestion_quality.py")
mod = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)  # type: ignore[union-attr]


def write(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def base(st, i):
    return {"doc_id": f"{st}:{i}", "source_type": st, "source_subtype": f"{st}_x", "title": st, "content": "This content is long enough for quality checks.", "metadata": {}}


def test_quality_passes_valid_small_fixture(tmp_path):
    p=tmp_path/"docs.jsonl"; write(p, [base("trademark",1), base("patent",1), base("litigation",1)])
    r=mod.check_quality(p)
    assert r["missing_required_source_types"] == [] and r["duplicate_doc_id_count"] == 0


def test_quality_detects_missing_required_source_type(tmp_path):
    p=tmp_path/"docs.jsonl"; write(p, [base("trademark",1)])
    assert "patent" in mod.check_quality(p)["missing_required_source_types"]


def test_quality_detects_duplicate_doc_id(tmp_path):
    p=tmp_path/"docs.jsonl"; a=base("trademark",1); b=base("trademark",2); b["doc_id"]=a["doc_id"]; write(p,[a,b,base("patent",1),base("litigation",1)])
    assert mod.check_quality(p)["duplicate_doc_id_count"] == 1


def test_quality_detects_malformed_metadata_json(tmp_path):
    p=tmp_path/"docs.jsonl"; r=base("trademark",1); r["metadata_json"]="{"; write(p,[r,base("patent",1),base("litigation",1)])
    assert mod.check_quality(p)["metadata_json_parse_failures"] == 1


def test_quality_help_works():
    cp=subprocess.run([sys.executable, str(ROOT/"scripts"/"check_ingestion_quality.py"), "--help"], text=True, capture_output=True)
    assert cp.returncode == 0 and "--require-source-types" in cp.stdout
