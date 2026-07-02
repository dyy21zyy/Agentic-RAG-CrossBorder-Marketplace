"""Source-balanced retrieval for mixed and risk queries."""
from __future__ import annotations

from copy import copy
from typing import Any

from crossborder_agentic_rag.schemas.evidence import EvidenceChunk

DEFAULT_SOURCE_TYPES = ["trademark", "patent", "litigation"]


class SourceBalancedRetriever:
    """Retrieve from each requested source independently, then merge results.

    This wrapper keeps broad mixed/risk queries from being dominated by one
    evidence source by delegating one retrieval call per source type to an
    existing retriever.
    """

    def __init__(
        self,
        base_retriever: Any,
        mode: str = "hybrid_rerank",
        per_source_k: int = 10,
        final_k: int = 20,
        candidate_k: int = 50,
    ) -> None:
        self.base_retriever = base_retriever
        self.mode = mode
        self.per_source_k = per_source_k
        self.final_k = final_k
        self.candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        dense_vector: list[float] | None = None,
        filters: dict | None = None,
        source_types: list[str] | None = None,
    ) -> list[EvidenceChunk]:
        selected_sources = source_types or list(DEFAULT_SOURCE_TYPES)
        if self.per_source_k <= 0 or self.final_k <= 0 or not selected_sources:
            return []

        retrieval_filters = dict(filters or {})
        retrieval_filters.pop("source_type", None)

        by_chunk_id: dict[str, EvidenceChunk] = {}
        for source_type in selected_sources:
            try:
                results = self.base_retriever.retrieve(
                    query=query,
                    dense_vector=dense_vector,
                    filters=retrieval_filters,
                    top_k=self.per_source_k,
                    source_types=[source_type],
                    mode=self.mode,
                    candidate_k=self.candidate_k,
                )
            except TypeError as exc:
                if "candidate_k" not in str(exc):
                    raise
                results = self.base_retriever.retrieve(
                    query=query,
                    dense_vector=dense_vector,
                    filters=retrieval_filters,
                    top_k=self.per_source_k,
                    source_types=[source_type],
                    mode=self.mode,
                )
            for chunk in results:
                marked = copy(chunk)
                marked.metadata = dict(chunk.metadata)
                marked.metadata["source_balanced"] = True
                marked.metadata["retrieved_from_source"] = source_type
                existing = by_chunk_id.get(marked.chunk_id)
                if existing is None or marked.score > existing.score:
                    by_chunk_id[marked.chunk_id] = marked

        return sorted(by_chunk_id.values(), key=lambda chunk: chunk.score, reverse=True)[: self.final_k]
