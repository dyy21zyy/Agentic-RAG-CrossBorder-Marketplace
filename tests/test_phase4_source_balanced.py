from __future__ import annotations

from crossborder_agentic_rag.retrieval.source_balanced import SourceBalancedRetriever
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk


def chunk(source_type: str, suffix: str, score: float = 1.0) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=f"{source_type}-{suffix}",
        doc_id=f"doc-{source_type}-{suffix}",
        source_type=source_type,
        source_subtype="record",
        title=f"{source_type} title",
        content=f"{source_type} content",
        score=score,
    )


class FakeRetriever:
    def __init__(self):
        self.calls = []
        self.results = {
            "trademark": [chunk("trademark", "1", 0.8)],
            "patent": [chunk("patent", "1", 0.9)],
            "litigation": [chunk("litigation", "1", 0.7)],
        }

    def retrieve(self, query, dense_vector=None, filters=None, top_k=20, source_types=None, mode="hybrid_rrf", candidate_k=None):
        self.calls.append({"query": query, "filters": filters, "source_types": source_types, "top_k": top_k, "mode": mode, "candidate_k": candidate_k})
        source_type = source_types[0]
        return self.results[source_type][:top_k]


def test_source_balanced_retrieves_multiple_sources():
    fake = FakeRetriever()
    results = SourceBalancedRetriever(fake, per_source_k=2, final_k=10).retrieve("risk", source_types=["trademark", "patent", "litigation"])
    assert {result.source_type for result in results} == {"trademark", "patent", "litigation"}
    assert [call["source_types"] for call in fake.calls] == [["trademark"], ["patent"], ["litigation"]]
    assert all(result.metadata["source_balanced"] is True for result in results)
    assert {result.metadata["retrieved_from_source"] for result in results} == {"trademark", "patent", "litigation"}


def test_source_balanced_single_source_only_retrieves_that_source():
    fake = FakeRetriever()
    results = SourceBalancedRetriever(fake).retrieve("risk", source_types=["patent"])
    assert [result.source_type for result in results] == ["patent"]
    assert [call["source_types"] for call in fake.calls] == [["patent"]]


def test_source_balanced_defaults_to_all_three_sources_and_removes_filter_conflict():
    fake = FakeRetriever()
    results = SourceBalancedRetriever(fake).retrieve("risk", filters={"source_type": "patent", "word_mark": "SMARTPACK"}, source_types=None)
    assert {result.source_type for result in results} == {"trademark", "patent", "litigation"}
    assert [call["source_types"] for call in fake.calls] == [["trademark"], ["patent"], ["litigation"]]
    assert all(call["filters"] == {"word_mark": "SMARTPACK"} for call in fake.calls)
