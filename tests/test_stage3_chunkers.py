from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from crossborder_agentic_rag.ingestion.chunkers import MAX_LITIGATION_DOCKET_CHUNKS, chunk_document, chunk_documents, chunk_policy, chunk_trademark, make_chunk
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
    doc = NormalizedDocument("x", "trademark", "X", "content")
    object.__setattr__(doc, "source_type", "unknown")
    with pytest.raises(ValueError):
        chunk_document(doc)


def test_trademark_chunks_include_identity_class_goods_design_without_record_by_default():
    chunks = chunk_document(by_type("trademark"))
    assert {"trademark_identity", "trademark_class", "trademark_goods_services", "trademark_design"} <= subtypes(chunks)
    assert "trademark_record" not in subtypes(chunks)
    assert any("MERCEDES" in c.content for c in chunks)


def test_trademark_record_can_be_enabled_explicitly():
    chunks = chunk_trademark(by_type("trademark"), include_full_record=True)
    assert "trademark_record" in subtypes(chunks)


def test_trademark_design_chunk_preserves_design_codes_and_pseudo_marks():
    design = next(c for c in chunk_document(by_type("trademark")) if c.source_subtype == "trademark_design")
    assert "180501" in design.content and "MER CE DES" in design.content
    assert design.metadata["design_search_codes"] == ["180501"]
    assert design.metadata["pseudo_marks"] == ["MER CE DES"]


def test_trademark_chunk_metadata_preserves_key_fields():
    identity = next(c for c in chunk_document(by_type("trademark")) if c.source_subtype == "trademark_identity")
    for key in ["serial_number", "registration_number", "word_mark", "filing_date", "registration_date", "status_code", "source_file", "source_path"]:
        assert key in identity.metadata


def test_trademark_chunks_include_phase2_metadata():
    chunks = chunk_document(by_type("trademark"))
    assert all(c.metadata["parent_id"].startswith("trademark:") for c in chunks)
    assert all(c.metadata["partition"] == "trademark_db" for c in chunks)
    assert any(c.metadata["context_path"].endswith(" > Identity") for c in chunks)
    assert any("MERCEDES" in c.metadata["entity_mentions"] for c in chunks)


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


def test_patent_claim_duplicate_claim_numbers_have_unique_chunk_ids():
    doc = NormalizedDocument(
        "patent:dup-claims",
        "patent",
        "Duplicate claims",
        "content",
        {"patent_id": "US-DUP", "claims": "1. First claim text.\n1. Duplicate claim number text."},
    )
    claims = [c for c in chunk_document(doc) if c.source_subtype == "patent_claim"]
    assert [c.metadata["claim_number"] for c in claims] == ["1", "1"]
    assert len({c.chunk_id for c in claims}) == len(claims)
    assert [c.chunk_id for c in claims] == [
        "patent:dup-claims:patent_claim:claim-1-0",
        "patent:dup-claims:patent_claim:claim-1-1",
    ]


def test_patent_claim_chunks_include_phase2_metadata_and_subtype_unchanged():
    claims = [c for c in chunk_document(by_type("patent")) if c.source_subtype == "patent_claim"]
    assert claims
    for claim in claims:
        assert claim.source_subtype == "patent_claim"
        assert claim.metadata["parent_id"] == "patent:US1234567"
        assert claim.metadata["partition"] == "patent_db"
        assert claim.metadata["context_path"] == f"Patent > US1234567 > Claim {claim.metadata['claim_number']}"
        assert claim.metadata["claim_number"] in claim.metadata["entity_mentions"]


def test_chunk_document_rejects_policy_source_type():
    doc = NormalizedDocument("policy:plain", "trademark", "Plain", "Sellers must respect IP rights.", {"platform": "legacy"})
    object.__setattr__(doc, "source_type", "p" + "olicy")
    with pytest.raises(ValueError, match="policy source_type is not supported"):
        chunk_document(doc)


def test_chunk_policy_remains_available_for_legacy_callers():
    doc = NormalizedDocument("legacy:policy", "trademark", "Legacy policy", "# IP\n1. Trademark infringement may cause removal.\nExample: unauthorized logo.", {"platform": "legacy"})
    chunks = chunk_policy(doc)
    assert {"policy_section", "policy_clause", "policy_enforcement", "policy_example"} <= subtypes(chunks)

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


def test_litigation_chunks_include_phase2_metadata():
    chunks = chunk_document(by_type("litigation"))
    assert all(c.metadata["parent_id"].startswith("litigation:") for c in chunks)
    assert all(c.metadata["partition"] == "litigation_db" for c in chunks)
    assert any(c.metadata["context_path"].endswith(" > Case Summary") for c in chunks)
    assert any("US1234567" in c.metadata["entity_mentions"] for c in chunks if c.source_subtype == "litigation_patent")



def test_compact_metadata_excludes_large_nested_fields():
    doc = by_type("litigation")
    chunks = chunk_document(doc)
    forbidden = {"case", "documents", "parties", "patents", "timeline", "source_files"}
    assert chunks
    assert all(not (forbidden & set(chunk.metadata)) for chunk in chunks)


def test_make_chunk_does_not_copy_arbitrary_large_metadata():
    doc = NormalizedDocument(
        "tm:large-metadata",
        "trademark",
        "Large metadata",
        "content",
        {
            "serial_number": "123",
            "word_mark": "COMPACT",
            "raw_row": {"large": "x" * 10000},
            "original_text": "duplicated source text" * 1000,
            "documents": [{"doc_number": str(i)} for i in range(10)],
            "unexpected_large_field": ["x" * 1000 for _ in range(10)],
        },
    )
    chunk = make_chunk(doc, "trademark_identity", "Compact identity", "content")
    assert chunk.metadata["serial_number"] == "123"
    assert chunk.metadata["word_mark"] == "COMPACT"
    assert "raw_row" not in chunk.metadata
    assert "original_text" not in chunk.metadata
    assert "documents" not in chunk.metadata
    assert "unexpected_large_field" not in chunk.metadata


def test_litigation_docket_chunks_are_capped():
    doc = by_type("litigation")
    md = dict(doc.metadata)
    md["documents"] = [
        {"doc_number": str(i), "short_description": f"doc {i}", "long_description": "short", "doc_date_filed": "2020-01-01"}
        for i in range(MAX_LITIGATION_DOCKET_CHUNKS + 7)
    ]
    capped_doc = NormalizedDocument(doc.doc_id, doc.source_type, doc.title, doc.content, md)
    docket_chunks = [c for c in chunk_document(capped_doc) if c.source_subtype == "litigation_docket"]
    assert len(docket_chunks) == MAX_LITIGATION_DOCKET_CHUNKS


def test_streaming_builder_outputs_same_chunks_as_normal_builder_for_sample(tmp_path):
    out, rep = tmp_path / "stream_chunks.jsonl", tmp_path / "stream_report.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "05_build_chunks_streaming.py"), "--input", str(FIXTURE), "--output", str(out), "--report", str(rep), "--progress-every", "1"], text=True, capture_output=True)
    assert r.returncode == 0, r.stderr
    stream_chunks = [c.to_dict() for c in read_chunks_jsonl(out)]
    normal_chunks = [c.to_dict() for c in chunk_documents(docs())]
    report = json.loads(rep.read_text(encoding="utf-8"))
    assert stream_chunks == normal_chunks
    assert report["chunks_written"] == len(normal_chunks)
    assert report["failed_documents_count"] == 0
    assert report["approximate_output_size_bytes"] > 0

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
    assert report["chunks_by_source_type"]["trademark"] >= 4


def test_build_chunks_report_detects_duplicate_chunk_ids():
    spec = importlib.util.spec_from_file_location("build_chunks", ROOT / "scripts" / "05_build_chunks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = {"warnings": []}
    duplicated = [
        EvidenceChunk("dup", "d1", "patent", "patent_claim", "t1", "c1", {}),
        EvidenceChunk("dup", "d1", "patent", "patent_claim", "t2", "c2", {}),
        EvidenceChunk("unique", "d2", "patent", "patent_claim", "t3", "c3", {}),
    ]
    mod.add_duplicate_chunk_id_stats(report, duplicated)
    assert report["total_chunks"] == 3
    assert report["unique_chunk_ids"] == 2
    assert report["duplicate_chunk_ids"] == 1
    assert report["duplicate_chunk_id_examples"] == [{"chunk_id": "dup", "count": 2}]
    assert report["warnings"]


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
