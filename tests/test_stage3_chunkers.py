from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from crossborder_agentic_rag.ingestion.chunkers import chunk_document, chunk_documents
from crossborder_agentic_rag.ingestion.io_utils import read_chunks_jsonl, read_documents_jsonl, write_chunks_jsonl
from crossborder_agentic_rag.schemas.documents import NormalizedDocument
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "chunks" / "sample_normalized_docs.jsonl"


def docs():
    return read_documents_jsonl(FIXTURE)


def by_type(source_type: str):
    return next(d for d in docs() if d.source_type == source_type)


def subtypes(chunks):
    return {c.source_subtype for c in chunks}


def assert_valid_chunks(chunks, doc):
    assert chunks
    assert all(isinstance(c, EvidenceChunk) for c in chunks)
    assert all(c.chunk_id and c.doc_id == doc.doc_id and c.source_type == doc.source_type for c in chunks)
    assert all(c.content.strip() and c.score == 0.0 for c in chunks)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunk_document(doc)]


def test_chunk_document_dispatches_by_source_type():
    for doc in docs():
        assert_valid_chunks(chunk_document(doc), doc)
    assert len(chunk_documents(docs())) >= 10


def test_chunk_document_rejects_unknown_source_type():
    doc = NormalizedDocument("x", "policy", "X", "content")
    object.__setattr__(doc, "source_type", "unknown")
    with pytest.raises(ValueError):
        chunk_document(doc)


def test_trademark_chunks_include_identity_class_goods_design_record():
    chunks = chunk_document(by_type("trademark"))
    assert {"trademark_identity", "trademark_class", "trademark_goods_services", "trademark_design", "trademark_record"} <= subtypes(chunks)
    assert any("MERCEDES" in c.content for c in chunks)


def test_trademark_design_chunk_preserves_design_codes_and_pseudo_marks():
    design = next(c for c in chunk_document(by_type("trademark")) if c.source_subtype == "trademark_design")
    assert "180501" in design.content and "MER CE DES" in design.content
    assert design.metadata["design_search_codes"] == ["180501"]
    assert design.metadata["pseudo_marks"] == ["MER CE DES"]


def test_trademark_chunk_metadata_preserves_key_fields():
    identity = next(c for c in chunk_document(by_type("trademark")) if c.source_subtype == "trademark_identity")
    for key in ["serial_number", "registration_number", "word_mark", "filing_date", "registration_date", "status_code", "source_file", "source_path"]:
        assert key in identity.metadata


def test_patent_claims_are_split_into_individual_chunks():
    claims = [c for c in chunk_document(by_type("patent")) if c.source_subtype == "patent_claim"]
    assert len(claims) == 2
    assert "Claim 1" in claims[0].content and "Claim 2" in claims[1].content


def test_patent_summary_detail_drawing_are_separate_chunks():
    chunks = chunk_document(by_type("patent"))
    assert {"patent_specification_summary", "patent_specification_detail", "patent_drawing"} <= subtypes(chunks)
    assert not any(c.source_subtype == "patent_claim" and "Figure 1" in c.content for c in chunks)


def test_patent_claim_chunks_preserve_claim_number():
    claims = [c for c in chunk_document(by_type("patent")) if c.source_subtype == "patent_claim"]
    assert [c.metadata["claim_number"] for c in claims] == ["1", "2"]
    assert all(c.metadata["patent_id"] == "US1234567" for c in claims)


def test_policy_chunks_include_section_clause_enforcement_example():
    chunks = chunk_document(by_type("policy"))
    assert {"policy_section", "policy_clause", "policy_enforcement", "policy_example"} <= subtypes(chunks)


def test_policy_enforcement_chunk_detects_trademark_infringement_and_removal():
    enf = [c for c in chunk_document(by_type("policy")) if c.source_subtype == "policy_enforcement"]
    joined = "\n".join(c.content.lower() for c in enf)
    assert "trademark infringement" in joined and "remove" in joined or "removal" in joined


def test_policy_fallback_section_when_no_headings():
    doc = NormalizedDocument("policy:plain", "policy", "Plain", "Sellers must respect IP rights.", {"platform": "Temu"})
    chunks = chunk_document(doc)
    assert len(chunks) == 1 and chunks[0].source_subtype == "policy_section"


def test_litigation_chunks_include_case_party_patent_docket_timeline():
    chunks = chunk_document(by_type("litigation"))
    assert {"litigation_case_summary", "litigation_party", "litigation_patent", "litigation_docket", "litigation_timeline"} <= subtypes(chunks)


def test_litigation_chunks_do_not_duplicate_full_case_content_for_every_chunk():
    chunks = chunk_document(by_type("litigation"))
    contents = [c.content for c in chunks]
    assert len(set(contents)) > 1
    assert not all("Documents include complaint and order" in c for c in contents)


def test_litigation_patent_chunk_preserves_patent_number():
    pat = next(c for c in chunk_document(by_type("litigation")) if c.source_subtype == "litigation_patent")
    assert pat.metadata["patent_number"] == "US1234567" and "US1234567" in pat.content


def test_write_and_read_chunks_jsonl_roundtrip(tmp_path):
    chunks = chunk_documents(docs())
    out = tmp_path / "chunks.jsonl"
    assert write_chunks_jsonl(chunks, out) == len(chunks)
    assert [c.to_dict() for c in read_chunks_jsonl(out)] == [c.to_dict() for c in chunks]


def test_build_chunks_script_outputs_jsonl_and_report(tmp_path):
    out, rep = tmp_path / "chunks.jsonl", tmp_path / "report.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "05_build_chunks.py"), "--input", str(FIXTURE), "--output", str(out), "--report", str(rep)], text=True, capture_output=True)
    assert r.returncode == 0, r.stderr
    chunks = read_chunks_jsonl(out)
    report = json.loads(rep.read_text(encoding="utf-8"))
    assert chunks and report["documents_seen"] == 4 and report["documents_chunked"] == 4
    assert report["chunks_written"] == len(chunks)
    assert report["chunks_by_source_type"]["trademark"] >= 5


def test_build_chunks_script_fails_on_empty_input_without_allow_empty(tmp_path):
    inp, out, rep = tmp_path / "empty.jsonl", tmp_path / "chunks.jsonl", tmp_path / "report.json"
    inp.write_text("", encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "05_build_chunks.py"), "--input", str(inp), "--output", str(out), "--report", str(rep)], text=True, capture_output=True)
    assert r.returncode != 0 and "empty" in r.stderr.lower()


def test_build_chunks_script_allows_empty_input_with_allow_empty(tmp_path):
    inp, out, rep = tmp_path / "empty.jsonl", tmp_path / "chunks.jsonl", tmp_path / "report.json"
    inp.write_text("", encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "05_build_chunks.py"), "--input", str(inp), "--output", str(out), "--report", str(rep), "--allow-empty"], text=True, capture_output=True)
    assert r.returncode == 0, r.stderr
    assert read_chunks_jsonl(out) == []
    assert json.loads(rep.read_text(encoding="utf-8"))["warnings"]


def test_stage3_script_no_longer_raises_not_implemented():
    assert "NotImplementedError" not in (ROOT / "scripts" / "05_build_chunks.py").read_text(encoding="utf-8")


def test_future_stage_scripts_still_raise_not_implemented():
    for name in ["09_run_eval.py", "10_run_ablation.py"]:
        assert "NotImplementedError" not in (ROOT / "scripts" / name).read_text(encoding="utf-8")

def test_no_duplicate_module_paths_created():
    forbidden = [ROOT / "src" / "crossborder_agentic_rag" / "chunking", ROOT / "src" / "crossborder_agentic_rag" / "chunkers", ROOT / "src" / "crossborder_agentic_rag" / "ingestion" / "chunker_v2.py", ROOT / "src" / "crossborder_agentic_rag" / "retrieval" / "chunker.py"]
    assert not [p for p in forbidden if p.exists()]
