"""Stage 1 behavior tests for core schemas and interfaces."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from crossborder_agentic_rag.constants import RETRIEVAL_ROUTES, SOURCE_TYPES
from crossborder_agentic_rag.llm import BaseEmbeddingProvider, FakeEmbeddingProvider, build_embedding_provider
from crossborder_agentic_rag.retrieval import BaseReranker, NoOpReranker, build_reranker
from crossborder_agentic_rag.schemas import AgentState, EvidenceChunk, NormalizedDocument, QueryPlan

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "crossborder_agentic_rag"


def test_normalized_document_accepts_valid_source_types() -> None:
    for source_type in SOURCE_TYPES:
        doc = NormalizedDocument("doc-1", source_type, "Title", "Content")
        assert doc.source_type == source_type


def test_normalized_document_rejects_invalid_source_type() -> None:
    with pytest.raises(ValueError, match="source_type"):
        NormalizedDocument("doc-1", "invalid", "Title", "Content")


def test_normalized_document_rejects_empty_doc_id() -> None:
    with pytest.raises(ValueError, match="doc_id"):
        NormalizedDocument("", "trademark", "Title", "Content")


def test_normalized_document_to_dict_from_dict_roundtrip() -> None:
    doc = NormalizedDocument("doc-1", "trademark", "Title", "Content", {"jurisdiction": "US"})
    assert NormalizedDocument.from_dict(doc.to_dict()) == doc


def test_evidence_chunk_accepts_valid_source_types() -> None:
    for source_type in SOURCE_TYPES:
        chunk = EvidenceChunk("chunk-1", "doc-1", source_type, "subtype", "Title", "Content")
        assert chunk.source_type == source_type


def test_evidence_chunk_rejects_invalid_source_type() -> None:
    with pytest.raises(ValueError, match="source_type"):
        EvidenceChunk("chunk-1", "doc-1", "invalid", "subtype", "Title", "Content")


def test_evidence_chunk_rejects_empty_chunk_id() -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        EvidenceChunk("", "doc-1", "trademark", "subtype", "Title", "Content")


def test_evidence_chunk_score_is_float() -> None:
    chunk = EvidenceChunk("chunk-1", "doc-1", "trademark", "subtype", "Title", "Content", score="0.25")
    assert chunk.score == 0.25
    assert isinstance(chunk.score, float)


def test_evidence_chunk_to_dict_from_dict_roundtrip() -> None:
    chunk = EvidenceChunk("chunk-1", "doc-1", "patent", "claim", "Title", "Content", {"page": 1}, 0.9)
    assert EvidenceChunk.from_dict(chunk.to_dict()) == chunk


def test_query_plan_accepts_valid_routes() -> None:
    for route in RETRIEVAL_ROUTES:
        plan = QueryPlan("query", "lookup", "fact", route, source_types=["trademark"])
        assert plan.retrieval_route == route


def test_query_plan_rejects_invalid_route() -> None:
    with pytest.raises(ValueError, match="retrieval_route"):
        QueryPlan("query", "lookup", "fact", "invalid")


def test_query_plan_rejects_invalid_source_type_filter() -> None:
    with pytest.raises(ValueError, match="source_types"):
        QueryPlan("query", "lookup", "fact", "hybrid", source_types=["invalid"])


def test_query_plan_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        QueryPlan("query", "lookup", "fact", "hybrid", top_k=0)


def test_query_plan_to_dict_from_dict_roundtrip() -> None:
    plan = QueryPlan("query", "lookup", "fact", "mixed", {"year": 2024}, ["trademark"], 5)
    assert QueryPlan.from_dict(plan.to_dict()) == plan


def test_agent_state_defaults_are_independent_lists() -> None:
    first = AgentState("query one")
    second = AgentState("query two")
    first.trace.append("step")
    first.tool_calls.append({"tool": "x", "payload": {}})
    assert second.trace == []
    assert second.tool_calls == []


def test_agent_state_add_trace() -> None:
    state = AgentState("query")
    state.add_trace("classified")
    assert state.trace == ["classified"]


def test_agent_state_add_tool_call() -> None:
    state = AgentState("query")
    state.add_tool_call("search", {"top_k": 5})
    assert state.tool_calls == [{"tool": "search", "payload": {"top_k": 5}}]


def test_fake_embedding_is_deterministic() -> None:
    provider = FakeEmbeddingProvider(dim=8)
    assert provider.embed_query("same text") == provider.embed_query("same text")
    assert provider.embed_query("same text") != provider.embed_query("different text")


def test_fake_embedding_has_expected_dimension() -> None:
    assert len(FakeEmbeddingProvider(dim=7).embed_query("text")) == 7


def test_fake_embedding_is_l2_normalized() -> None:
    vector = FakeEmbeddingProvider(dim=16).embed_query("normalized")
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


def test_fake_embedding_records_query_calls() -> None:
    provider = FakeEmbeddingProvider()
    provider.embed_query("q1")
    provider.embed_query("q2")
    assert provider.query_calls == ["q1", "q2"]


def test_fake_embedding_records_document_calls() -> None:
    provider = FakeEmbeddingProvider(dim=4)
    vectors = provider.embed_documents(["d1", "d2"])
    assert len(vectors) == 2
    assert provider.document_calls == ["d1", "d2"]
    assert provider.query_calls == []


def test_fake_embedding_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="dim"):
        FakeEmbeddingProvider(dim=0)


def test_embedding_factory_supports_fake() -> None:
    assert isinstance(build_embedding_provider("fake", dim=3), FakeEmbeddingProvider)


def test_embedding_factory_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    assert isinstance(build_embedding_provider(dim=3), FakeEmbeddingProvider)


def test_embedding_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Missing embedding API"):
        build_embedding_provider("openai")


def _chunk(identifier: int, score: float = 1.0) -> EvidenceChunk:
    return EvidenceChunk(f"chunk-{identifier}", f"doc-{identifier}", "trademark", "trademark", f"Title {identifier}", "Content", score=score)


def test_noop_reranker_returns_top_k() -> None:
    candidates = [_chunk(1), _chunk(2), _chunk(3)]
    assert NoOpReranker().rerank("query", candidates, 2) == candidates[:2]


def test_noop_reranker_preserves_candidate_order() -> None:
    candidates = [_chunk(1), _chunk(2), _chunk(3)]
    result = NoOpReranker().rerank("query", candidates, 3)
    assert result == candidates
    assert all(actual is expected for actual, expected in zip(result, candidates, strict=True))


def test_noop_reranker_handles_zero_top_k() -> None:
    assert NoOpReranker().rerank("query", [_chunk(1)], 0) == []


def test_reranker_factory_supports_noop() -> None:
    assert isinstance(build_reranker("noop"), NoOpReranker)


def test_reranker_factory_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER_PROVIDER", "noop")
    assert isinstance(build_reranker(), NoOpReranker)


def test_reranker_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ImportError, match="sentence-transformers"):
        build_reranker("cross_encoder")


def test_public_schema_exports() -> None:
    import crossborder_agentic_rag.schemas as schemas

    assert schemas.__all__ == ["NormalizedDocument", "EvidenceChunk", "QueryPlan", "AgentState"]


def test_public_embedding_exports() -> None:
    import crossborder_agentic_rag.llm as llm

    assert "BaseEmbeddingProvider" in llm.__all__ and "FakeEmbeddingProvider" in llm.__all__ and "build_embedding_provider" in llm.__all__
    assert issubclass(FakeEmbeddingProvider, BaseEmbeddingProvider)


def test_public_reranker_exports() -> None:
    import crossborder_agentic_rag.retrieval as retrieval

    assert "BaseReranker" in retrieval.__all__ and "NoOpReranker" in retrieval.__all__ and "build_reranker" in retrieval.__all__
    assert issubclass(NoOpReranker, BaseReranker)


def test_stage0_scripts_still_raise_not_implemented() -> None:
    """Scripts 01-10 should no longer be placeholders after Stage 7."""
    for script_path in sorted((ROOT / "scripts").glob("*.py")):
        assert "NotImplementedError" not in script_path.read_text(encoding="utf-8")
