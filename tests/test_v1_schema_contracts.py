import json
from datetime import datetime
from pathlib import Path

import pytest

from crossborder_agentic_rag.schemas import (
    EvidenceChunk,
    EvidenceHit,
    ImageAsset,
    NormalizedDocument,
    RiskScreeningReport,
    RiskVerdict,
    TraceEvent,
)
from crossborder_agentic_rag.ingestion.io_utils import read_chunks_jsonl


def _report_kwargs() -> dict:
    return {
        "report_id": "report-1",
        "trace_id": "trace-1",
        "created_at": "2026-08-03T00:00:00Z",
        "product_profile": {"name": "smart phone case"},
        "target_markets": ["US"],
        "screening_scope": ["trademark"],
        "overall_verdict": "caution",
        "country_summaries": [],
        "risk_cards": {},
        "module_results": [],
        "evidence_items": [],
        "action_recommendations": [],
        "missing_evidence": [],
        "limitations": [],
    }


def test_text_document_defaults_to_empty_images():
    doc = NormalizedDocument(
        doc_id="trademark:1",
        source_type="trademark",
        title="MARK",
        content="Goods and services evidence",
    )
    assert doc.images == []
    assert doc.to_dict()["images"] == []


def test_document_from_legacy_json_without_images():
    doc = NormalizedDocument.from_dict(
        {
            "doc_id": "patent:1",
            "source_type": "patent",
            "title": "Patent 1",
            "content": "Claim evidence",
            "metadata": {},
        }
    )
    assert doc.images == []


def test_read_fixture_chunks_have_images_field():
    chunks = read_chunks_jsonl(Path("tests/fixtures/agent/sample_chunks.jsonl"))
    assert chunks
    assert all(chunk.images == [] for chunk in chunks)


def test_risk_report_required_fields_round_trip():
    hit = EvidenceHit(
        evidence_id="E1",
        chunk_id="trademark:1:chunk:0",
        source_type="trademark",
        title="Trademark evidence",
        content="Registered mark evidence",
        citation="[trademark:1:chunk:0] Trademark evidence",
        rank=1,
        score=1.0,
        retrieval_mode="bm25_only",
        tool_name="trademark_search_tool",
    )
    report = RiskScreeningReport(
        report_id="report-1",
        trace_id="trace-1",
        created_at="2026-08-03T00:00:00Z",
        product_profile={"name": "smart phone case"},
        target_markets=["US"],
        screening_scope=["trademark"],
        overall_verdict=RiskVerdict.CAUTION,
        country_summaries=[{"country": "US", "verdict": "caution", "summary": "Trademark evidence requires review."}],
        risk_cards={"no_risk_found": 0, "caution": 1, "not_recommended": 0, "insufficient_evidence": 0},
        module_results=[{"module": "trademark", "verdict": "caution", "evidence_ids": ["E1"]}],
        evidence_items=[hit],
        action_recommendations=["人工复核商标证据后再决定是否上架。"],
        missing_evidence=[],
        limitations=["本报告仅用于知识产权风险初筛和证据发现，不构成法律意见。"],
    )
    assert report.to_dict()["overall_verdict"] == "caution"
    assert report.to_dict()["evidence_items"][0]["evidence_id"] == "E1"


def test_trace_event_has_json_serializable_payload():
    event = TraceEvent(
        trace_id="trace-1",
        step="planner",
        event_type="llm_plan",
        payload={"tool_count": 2},
        timestamp="2026-08-03T00:00:00Z",
    )
    assert event.to_dict()["payload"] == {"tool_count": 2}


def test_nested_non_json_values_are_rejected_before_serialization():
    trace = TraceEvent("trace-1", "planner", "llm_plan", {"nested": [{"bad": {1, 2}}]}, "2026-08-03T00:00:00Z")
    with pytest.raises(TypeError, match="JSON-serializable"):
        trace.to_dict()

    image = ImageAsset("image-1", "doc-1", metadata={"nested": {"created": datetime(2026, 8, 3)}})
    with pytest.raises(TypeError, match="JSON-serializable"):
        image.to_dict()

    hit = EvidenceHit(
            evidence_id="E1",
            chunk_id="trademark:1:chunk:0",
            source_type="trademark",
            title="Trademark evidence",
            content="Registered mark evidence",
            citation="[trademark:1:chunk:0] Trademark evidence",
            rank=1,
            score=1.0,
            retrieval_mode="bm25_only",
            tool_name="trademark_search_tool",
            metadata={"nested": {"bad": object()}},
        )
    with pytest.raises(TypeError, match="JSON-serializable"):
        hit.to_dict()

    report_kwargs = _report_kwargs()
    report_kwargs["product_profile"] = {"nested": {"bad": {"set"}}}
    report = RiskScreeningReport(**report_kwargs)
    with pytest.raises(TypeError, match="JSON-serializable"):
        report.to_dict()

    safe_event = TraceEvent("trace-1", "planner", "llm_plan", {"nested": [{"value": 1}]}, "2026-08-03T00:00:00Z")
    json.dumps(safe_event.to_dict())


def test_core_domains_verdicts_and_scopes_are_enforced():
    with pytest.raises(ValueError, match="source_type"):
        EvidenceHit(
            evidence_id="E1",
            chunk_id="unknown:1:chunk:0",
            source_type="copyright",
            title="Evidence",
            content="Content",
            citation="[unknown:1:chunk:0] Evidence",
            rank=1,
            score=1.0,
            retrieval_mode="bm25_only",
            tool_name="search_tool",
        )

    report_kwargs = _report_kwargs()
    report = RiskScreeningReport(**report_kwargs)
    assert report.overall_verdict is RiskVerdict.CAUTION

    invalid_scope = _report_kwargs()
    invalid_scope["screening_scope"] = ["trademark", "copyright"]
    with pytest.raises(ValueError, match="screening_scope"):
        RiskScreeningReport(**invalid_scope)

    invalid_verdict = _report_kwargs()
    invalid_verdict["overall_verdict"] = "infringement_found"
    with pytest.raises(ValueError, match="overall_verdict"):
        RiskScreeningReport(**invalid_verdict)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("action_recommendations", [{"bad"}]),
        ("missing_evidence", [{"bad"}]),
        ("limitations", [{"bad"}]),
        ("langfuse_url", {"bad"}),
    ],
)
def test_report_rejects_unsupported_nested_output_values(field_name, value):
    report_kwargs = _report_kwargs()
    report_kwargs[field_name] = value
    report = RiskScreeningReport(**report_kwargs)
    with pytest.raises(TypeError, match="JSON-serializable"):
        report.to_dict()
