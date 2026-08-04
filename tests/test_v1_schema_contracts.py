from crossborder_agentic_rag.schemas import (
    EvidenceChunk,
    EvidenceHit,
    ImageAsset,
    NormalizedDocument,
    RiskScreeningReport,
    RiskVerdict,
    TraceEvent,
)


def test_text_document_defaults_to_empty_images():
    doc = NormalizedDocument(
        doc_id="trademark:1",
        source_type="trademark",
        title="MARK",
        content="Goods and services evidence",
    )
    assert doc.images == []
    assert doc.to_dict()["images"] == []


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
