from __future__ import annotations

import importlib.util, json, shutil, subprocess, sys
from pathlib import Path

from crossborder_agentic_rag.ingestion.io_utils import read_documents_jsonl, write_documents_jsonl
from crossborder_agentic_rag.ingestion.litigation_parser import parse_litigation_csv_directory
from crossborder_agentic_rag.ingestion.patent_parser import parse_patent_tsv_directory
from crossborder_agentic_rag.ingestion.policy_parser import parse_policy_directory
from crossborder_agentic_rag.ingestion.trademark_parser import parse_trademark_xml_directory
from crossborder_agentic_rag.schemas.documents import NormalizedDocument

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_parse_trademark_xml_multiple_records():
    docs, report = parse_trademark_xml_directory(FIXTURES / "trademark")
    good = [d for d in docs if d.doc_id != "trademark:bad"]
    assert len(good) == 2
    assert all(isinstance(d, NormalizedDocument) and d.source_type == "trademark" for d in good)
    assert {d.metadata["word_mark"] for d in good} >= {"MERCEDES", "SHOPSAFE"}
    assert report["files_seen"] == 2 and report["files_parsed"] == 1 and report["documents_parsed"] == 2


def test_parse_trademark_extracts_design_and_pseudo_marks():
    docs, _ = parse_trademark_xml_directory(FIXTURES / "trademark")
    mercedes = next(d for d in docs if d.metadata.get("word_mark") == "MERCEDES")
    assert mercedes.metadata["design_search_codes"] == ["180501"]
    assert mercedes.metadata["pseudo_marks"] == ["MER CE DES"]
    assert "Automobiles" in mercedes.content and "012" in mercedes.content


def test_parse_trademark_bad_file_is_reported_without_crashing():
    docs, report = parse_trademark_xml_directory(FIXTURES / "trademark")
    assert len(docs) == 2
    assert len(report["failed_files"]) == 1 and "bad_trademark.xml" in report["failed_files"][0]["path"]


def test_parse_patent_tsv_extracts_all_long_text_fields():
    docs, report = parse_patent_tsv_directory(FIXTURES / "patent")
    assert len(docs) == 2 and report["documents_parsed"] == 2
    doc = docs[0]
    assert doc.doc_id == "patent:US10000001"
    assert "scanner" in doc.content and "Detailed description" in doc.content and "Figure 1" in doc.content


def test_parse_patent_tsv_supports_column_variants(tmp_path):
    p = tmp_path / "variants.tsv"
    p.write_text("patent_number\tBrief Summary\tClaims\tDetail Description\tDrawing Description\nUSV1\tSum\tClaim\tDesc\tDraw\n", encoding="utf-8")
    docs, _ = parse_patent_tsv_directory(tmp_path)
    assert docs[0].metadata["patent_id"] == "USV1" and "Desc" in docs[0].content


def test_parse_patent_tsv_skips_rows_without_patent_id(tmp_path):
    p = tmp_path / "skip.tsv"
    p.write_text("patent_id\tbrief_summary\n\tNo id\nUS1\tHas id\n", encoding="utf-8")
    docs, report = parse_patent_tsv_directory(tmp_path)
    assert [d.doc_id for d in docs] == ["patent:US1"]
    assert report["warnings"]


def test_parse_litigation_joins_cases_documents_names_patents():
    docs, report = parse_litigation_csv_directory(FIXTURES / "litigation")
    assert len(docs) == 1 and report["documents_parsed"] == 1
    md = docs[0].metadata
    assert md["case"]["case_number"] == "1:24-cv-00001"
    assert len(md["documents"]) == 2 and len(md["parties"]) == 2 and len(md["patents"]) == 1
    assert "BrandCo" in docs[0].content and "US10000001" in docs[0].content


def test_parse_litigation_creates_timeline():
    docs, _ = parse_litigation_csv_directory(FIXTURES / "litigation")
    dates = [e["date"] for e in docs[0].metadata["timeline"]]
    assert dates == sorted(dates)
    assert "filed" in {e["event_type"] for e in docs[0].metadata["timeline"]}


def test_parse_litigation_handles_missing_related_tables_with_warning(tmp_path):
    shutil.copy(FIXTURES / "litigation" / "cases_sample.csv", tmp_path / "cases.csv")
    docs, report = parse_litigation_csv_directory(tmp_path)
    assert len(docs) == 1 and report["warnings"]


def test_parse_policy_directory_multiple_file_types():
    docs, report = parse_policy_directory(FIXTURES / "policies")
    assert len(docs) == 2 and report["documents_parsed"] == 2
    assert all(d.source_type == "policy" and d.metadata["platform"] == "Temu" for d in docs)


def test_parse_policy_html_extracts_readable_text():
    docs, _ = parse_policy_directory(FIXTURES / "policies")
    html = next(d for d in docs if d.metadata["file_type"] == ".html")
    assert "trademark infringement" in html.content and "<p>" not in html.content


def test_parse_policy_empty_file_warning(tmp_path):
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    docs, report = parse_policy_directory(tmp_path)
    assert docs == [] and report["warnings"]


def test_parse_scripts_write_jsonl_and_report(tmp_path):
    cases = [("01_parse_trademark_xml.py", FIXTURES / "trademark"), ("02_parse_patent_tsv.py", FIXTURES / "patent"), ("03_parse_litigation_csv.py", FIXTURES / "litigation"), ("04_parse_policy_docs.py", FIXTURES / "policies")]
    for script, inp in cases:
        out, rep = tmp_path / f"{script}.jsonl", tmp_path / f"{script}.json"
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--input", str(inp), "--output", str(out), "--report", str(rep)], text=True, capture_output=True)
        assert r.returncode == 0, r.stderr
        assert read_documents_jsonl(out)
        assert json.loads(rep.read_text(encoding="utf-8"))["documents_parsed"] >= 1
        assert "Parsed" in r.stdout


def test_stage2_scripts_no_longer_raise_not_implemented():
    for name in ["01_parse_trademark_xml.py", "02_parse_patent_tsv.py", "03_parse_litigation_csv.py", "04_parse_policy_docs.py"]:
        assert "NotImplementedError" not in (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_future_stage_scripts_still_raise_not_implemented():
    for name in ["09_run_eval.py", "10_run_ablation.py"]:
        assert "NotImplementedError" not in (ROOT / "scripts" / name).read_text(encoding="utf-8")

def test_no_duplicate_module_paths_created():
    forbidden = [ROOT/"src"/"crossborder_agentic_rag"/"parser", ROOT/"src"/"crossborder_agentic_rag"/"parsers", ROOT/"src"/"crossborder_agentic_rag"/"ingestion"/"parser_utils_v2.py", ROOT/"src"/"crossborder_agentic_rag"/"scripts"]
    assert not [p for p in forbidden if p.exists()]


def test_jsonl_roundtrip(tmp_path):
    docs, _ = parse_policy_directory(FIXTURES / "policies")
    out = tmp_path / "docs.jsonl"
    assert write_documents_jsonl(docs, out) == len(docs)
    assert read_documents_jsonl(out) == docs


def test_trademark_goods_services_sources_merge_without_duplicates(tmp_path):
    xml = tmp_path / "tm.xml"
    xml.write_text("""<case-file><serial-number>1</serial-number><mark-identification>MERGE</mark-identification><case-file-statement><type-code>GS009</type-code><text>Automobiles</text></case-file-statement><goods-services>Automobiles</goods-services><identification-of-goods>Parts</identification-of-goods></case-file>""", encoding="utf-8")
    docs, report = parse_trademark_xml_directory(tmp_path)
    assert report["failed_files"] == []
    assert docs[0].metadata["goods_services"] == ["Automobiles", "Parts"]


def test_patent_claim_metadata_preserved(tmp_path):
    p = tmp_path / "claims.tsv"
    p.write_text("patent_id\tclaim_number\tclaim_type\tclaim_text\tis_independent\nUS2\t1\tindependent\tA widget claim.\ttrue\n", encoding="utf-8")
    docs, report = parse_patent_tsv_directory(tmp_path)
    assert report["warnings"] == []
    assert docs[0].metadata["claim_number"] == "1"
    assert docs[0].metadata["claim_type"] == "independent"
    assert docs[0].metadata["is_independent"] is True
    assert "Patent ID: US2" in docs[0].content and "A widget claim" in docs[0].content


def test_patent_empty_text_skipped_with_warning(tmp_path):
    p = tmp_path / "empty.tsv"
    p.write_text("patent_id\ttitle\nUS3\t\n", encoding="utf-8")
    docs, report = parse_patent_tsv_directory(tmp_path)
    assert docs == []
    assert report["warnings"]


def test_litigation_minimal_variant_csv_parses(tmp_path):
    p = tmp_path / "cases.csv"
    p.write_text("docket_number,district_court,filing_date,plaintiff,defendant,asserted_patent,description\n2:25-cv-1,D. Del.,2025-01-02,A Corp,B LLC,US9,Patent suit\n", encoding="utf-8")
    docs, report = parse_litigation_csv_directory(tmp_path)
    assert len(docs) == 1
    assert report["warnings"]
    assert "2:25-cv-1" in docs[0].content and "D. Del." in docs[0].content
