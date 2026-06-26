"""Embedding provider interfaces.

Stage 1 includes only a deterministic fake provider for tests and smoke runs.
Stage 5 will add production providers.
"""

from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document strings."""
        return [self.embed_query(text) for text in texts]


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """FakeEmbeddingProvider is only for tests and smoke runs.

    It is not a semantic embedding model.
    """

    def __init__(self, dim: int = 16) -> None:
        if not isinstance(dim, int):
            raise TypeError("dim must be an integer")
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        self.query_calls: list[str] = []
        self.document_calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self.query_calls.append(text)
        return self._embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")
        self.document_calls.extend(texts)
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        values: list[float] = []
        counter = 0
        while len(values) < self.dim:
            digest = hashlib.sha256(f"{text}\0{counter}".encode("utf-8")).digest()
            for byte in digest:
                values.append((byte / 127.5) - 1.0)
                if len(values) == self.dim:
                    break
            counter += 1
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            return [0.0 for _ in values]
        return [value / norm for value in values]


def build_embedding_provider(provider: str | None = None, dim: int | None = None) -> BaseEmbeddingProvider:
    """Build a Stage 1 embedding provider."""
    provider_name = provider if provider is not None else os.getenv("EMBEDDING_PROVIDER", "fake")
    provider_name = provider_name.lower()
    if provider_name == "fake":
        return FakeEmbeddingProvider(dim=16 if dim is None else dim)
    raise NotImplementedError(
        f"Embedding provider '{provider_name}' is implemented in a later stage or not supported."
    )
