"""Reranking interface and Stage 1 no-op implementation."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from crossborder_agentic_rag.schemas.evidence import EvidenceChunk


class BaseReranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[EvidenceChunk],
        top_k: int,
    ) -> list[EvidenceChunk]:
        """Return reranked evidence candidates."""


class NoOpReranker(BaseReranker):
    """Reranker that preserves input order and scores."""

    def rerank(
        self,
        query: str,
        candidates: list[EvidenceChunk],
        top_k: int,
    ) -> list[EvidenceChunk]:
        if top_k <= 0:
            return []
        return candidates[:top_k]


def build_reranker(provider: str | None = None) -> BaseReranker:
    """Build a Stage 1 reranker."""
    provider_name = provider if provider is not None else os.getenv("RERANKER_PROVIDER", "noop")
    provider_name = provider_name.lower()
    if provider_name == "noop":
        return NoOpReranker()
    raise NotImplementedError(
        f"Reranker provider '{provider_name}' is implemented in a later stage or not supported."
    )
