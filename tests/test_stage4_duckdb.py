from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from crossborder_agentic_rag.ingestion.io_utils import read_documents_jsonl
from crossborder_agentic_rag.schemas.documents import NormalizedDocument
from crossborder_agentic_rag.storage.duckdb_store import DuckDBStore
from crossborder_agentic_rag.storage.schemas import REQUIRED_TABLES

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "duckdb" / "sample_normalized_docs.jsonl"


def docs():
    return read_documents_jsonl(FIXTURE)


@pytest.fixture()
def loaded_store():
    store = DuckDBStore(":memory:")
    report = store.load_documents(docs())
    yield store, report
    store.close()


def test_duckdb_creates_required_tables():
    store = DuckDBStore(":memory:"); store.initialize_schema()
    rows = store.connect().execute("show tables").fetchall()
    assert REQUIRED_TABLES <= {r[0] for r in rows}
    store.close()


def test_duckdb_loads_trademark_rows(loaded_store):
    store, report = loaded_store
    assert report["rows_inserted"]["trademarks"] == 1
    assert store.row_counts()["trademarks"] == 1


def test_duckdb_loads_trademark_classes(loaded_store):
    store, _ = loaded_store
    rows = store.lookup_trademark_classes_by_word_mark("MERCEDES")
    assert {r["nice_class"] for r in rows} == {"12", "28"}


def test_duckdb_loads_trademark_goods_services(loaded_store):
    store, _ = loaded_store
    rows = store.lookup_trademark_goods_services_by_word_mark("MERCEDES")
    assert any("Toy vehicles" in r["goods_services"] for r in rows)


def test_duckdb_lookup_trademark_by_registration_number(loaded_store):
    store, _ = loaded_store
    rows = store.lookup_trademark_by_registration_number("7000001")
    assert rows[0]["word_mark"] == "MERCEDES"


def test_duckdb_lookup_trademark_by_word_mark_case_insensitive(loaded_store):
    store, _ = loaded_store
    assert store.lookup_trademark_by_word_mark("mercedes")[0]["registration_number"] == "7000001"


def test_duckdb_lookup_trademark_classes_by_word_mark(loaded_store):
    store, _ = loaded_store
    assert len(store.lookup_trademark_classes_by_word_mark("mercedes")) == 2


def test_duckdb_lookup_trademark_goods_services_by_word_mark(loaded_store):
    store, _ = loaded_store
    assert len(store.lookup_trademark_goods_services_by_word_mark("mercedes")) == 2


def test_duckdb_loads_patent_rows(loaded_store):
    store, report = loaded_store
    assert report["rows_inserted"]["patents"] == 1
    assert store.row_counts()["patents"] == 1


def test_duckdb_lookup_patent_by_id(loaded_store):
    store, _ = loaded_store
    assert "counterfeit" in store.lookup_patent_by_id("US1234567")[0]["brief_summary"]


def test_duckdb_lookup_patent_by_patent_number_alias(loaded_store):
    store, _ = loaded_store
    assert store.lookup_patent_by_id("1234567")[0]["patent_id"] == "US1234567"


def test_duckdb_loads_litigation_case_rows(loaded_store):
    store, _ = loaded_store
    assert store.row_counts()["litigation_cases"] == 1


def test_duckdb_loads_litigation_documents(loaded_store):
    store, _ = loaded_store
    assert store.row_counts()["litigation_documents"] == 2


def test_duckdb_loads_litigation_names(loaded_store):
    store, _ = loaded_store
    assert store.row_counts()["litigation_names"] == 2


def test_duckdb_loads_litigation_patents(loaded_store):
    store, _ = loaded_store
    assert store.row_counts()["litigation_patents"] == 1


def test_duckdb_lookup_litigation_by_patent(loaded_store):
    store, _ = loaded_store
    rows = store.lookup_litigation_by_patent("US1234567")
    assert rows[0]["case_number"] == "1:24-cv-00001" and rows[0]["patent_number"] == "1234567"


def test_duckdb_lookup_litigation_by_case(loaded_store):
    store, _ = loaded_store
    assert store.lookup_litigation_by_case("1:24-cv-00001")[0]["court_name"] == "District of Delaware"


def test_duckdb_lookup_litigation_parties_by_case(loaded_store):
    store, _ = loaded_store
    assert {r["party_type"] for r in store.lookup_litigation_parties_by_case("1:24-cv-00001")} == {"plaintiff", "defendant"}


def test_duckdb_lookup_litigation_documents_by_case(loaded_store):
    store, _ = loaded_store
    assert {r["doc_number"] for r in store.lookup_litigation_documents_by_case("1:24-cv-00001")} == {"1", "12"}


def test_duckdb_lookup_litigation_patents_by_case(loaded_store):
    store, _ = loaded_store
    assert store.lookup_litigation_patents_by_case("1:24-cv-00001")[0]["patent"] == "US1234567"


def test_duckdb_skips_policy_docs_with_warning(loaded_store):
    _, report = loaded_store
    assert report["documents_skipped"] == 1
    assert any("policy" in w.lower() for w in report["warnings"])


def test_duckdb_row_counts_returns_all_tables(loaded_store):
    store, _ = loaded_store
    assert REQUIRED_TABLES <= set(store.row_counts())


def test_duckdb_load_report_contains_counts_warnings_and_failures():
    good = docs()[0]
    bad = NormalizedDocument("bad:patent", "patent", "Bad", "", {"patent_id": object()})
    store = DuckDBStore(":memory:")
    report = store.load_documents([good, bad, docs()[-1]])
    assert report["documents_seen"] == 3 and report["documents_loaded"] == 1
    assert report["warnings"] and report["failed_documents"]
    store.close()


def test_duckdb_returns_dicts_not_tuples(loaded_store):
    store, _ = loaded_store
    rows = store.lookup_trademark_by_word_mark("MERCEDES")
    assert isinstance(rows[0], dict)


def test_duckdb_close_is_idempotent():
    store = DuckDBStore(":memory:"); store.initialize_schema(); store.close(); store.close()


def test_build_duckdb_script_writes_database_and_report(tmp_path):
    db, rep = tmp_path / "ip.duckdb", tmp_path / "report.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "06_build_duckdb.py"), "--input", str(FIXTURE), "--duckdb-path", str(db), "--report", str(rep)], text=True, capture_output=True)
    assert r.returncode == 0, r.stderr
    data = json.loads(rep.read_text(encoding="utf-8"))
    assert db.exists() and data["row_counts"]["trademarks"] == 1 and data["documents_loaded"] == 3


def test_build_duckdb_script_fails_on_missing_input(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "06_build_duckdb.py"), "--input", str(tmp_path / "missing.jsonl"), "--duckdb-path", str(tmp_path / "x.duckdb"), "--report", str(tmp_path / "r.json")], text=True, capture_output=True)
    assert r.returncode != 0 and "does not exist" in r.stderr


def test_build_duckdb_script_fails_on_empty_input_without_allow_empty(tmp_path):
    inp = tmp_path / "empty.jsonl"; inp.write_text("", encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "06_build_duckdb.py"), "--input", str(inp), "--duckdb-path", str(tmp_path / "x.duckdb"), "--report", str(tmp_path / "r.json")], text=True, capture_output=True)
    assert r.returncode != 0 and "empty" in r.stderr.lower()


def test_build_duckdb_script_allows_empty_input_with_allow_empty(tmp_path):
    inp, db, rep = tmp_path / "empty.jsonl", tmp_path / "x.duckdb", tmp_path / "r.json"
    inp.write_text("", encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "06_build_duckdb.py"), "--input", str(inp), "--duckdb-path", str(db), "--report", str(rep), "--allow-empty"], text=True, capture_output=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(rep.read_text(encoding="utf-8"))["warnings"]


def test_build_duckdb_script_overwrite_rebuilds_database(tmp_path):
    db, rep = tmp_path / "ip.duckdb", tmp_path / "report.json"
    db.write_text("not duckdb", encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "06_build_duckdb.py"), "--input", str(FIXTURE), "--duckdb-path", str(db), "--report", str(rep), "--overwrite"], text=True, capture_output=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(rep.read_text(encoding="utf-8"))["row_counts"]["patents"] == 1


def test_stage4_script_no_longer_raises_not_implemented():
    assert "NotImplementedError" not in (ROOT / "scripts" / "06_build_duckdb.py").read_text(encoding="utf-8")


def test_future_stage_scripts_still_raise_not_implemented():
    for name in ["09_run_eval.py", "10_run_ablation.py"]:
        assert "NotImplementedError" not in (ROOT / "scripts" / name).read_text(encoding="utf-8")

def test_no_duplicate_module_paths_created():
    forbidden = [ROOT / "src" / "crossborder_agentic_rag" / "db", ROOT / "src" / "crossborder_agentic_rag" / "database", ROOT / "src" / "crossborder_agentic_rag" / "storage" / "duckdb.py", ROOT / "src" / "crossborder_agentic_rag" / "storage" / "duckdb_store_v2.py", ROOT / "src" / "crossborder_agentic_rag" / "storage" / "sql_store.py"]
    assert not [p for p in forbidden if p.exists()]
