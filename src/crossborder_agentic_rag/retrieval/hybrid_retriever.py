"""Hybrid retrieval orchestration."""
from __future__ import annotations
import os
from typing import Any
from crossborder_agentic_rag.llm.embeddings import BaseEmbeddingProvider
from crossborder_agentic_rag.retrieval.bm25 import LocalBM25Retriever
from crossborder_agentic_rag.retrieval.reranker import BaseReranker
from crossborder_agentic_rag.retrieval.rrf_fusion import rrf_fusion
from crossborder_agentic_rag.retrieval.utils import dedupe_chunks

class HybridRetriever:
    def __init__(self,embedding_provider:BaseEmbeddingProvider|None=None,bm25_retriever:LocalBM25Retriever|None=None,vector_store:Any|None=None,reranker:BaseReranker|None=None,rrf_k:int=60)->None:
        self.embedding_provider=embedding_provider; self.bm25_retriever=bm25_retriever; self.vector_store=vector_store; self.reranker=reranker; self.rrf_k=rrf_k
    def _bm25_exact_threshold(self) -> float:
        """Score threshold for BM25 exact identifier hits.

        Fast BM25 exact index gives large scores around 1000+.
        Normal lexical BM25 scores are much smaller.
        """
        try:
            return float(os.getenv("RAG_BM25_EXACT_SCORE_THRESHOLD", "900"))
        except Exception:
            return 900.0

    def _is_exact_bm25_hits(self, hits) -> bool:
        """Return True if BM25 hits look like exact identifier hits."""
        if not hits:
            return False
        score = getattr(hits[0], "score", 0.0)
        try:
            score = float(score or 0.0)
        except Exception:
            score = 0.0
        return score >= self._bm25_exact_threshold()

    def _maybe_exact_bm25_fast_path(self, query, filters, bm25_k, source_types):
        """Use BM25 exact index to avoid slow dense/Milvus for exact ID queries.

        This does not force the agent to select a tool. It only accelerates
        hybrid retrieval when the sparse retriever has already found exact
        identifier evidence.
        """
        try:
            hits = self._bm25(query, filters, bm25_k, source_types)
        except Exception:
            return None

        if self._is_exact_bm25_hits(hits):
            return hits
        return None

    def _bm25(self,q,filters,top_k,source_types):
        if self.bm25_retriever is None: raise ValueError("BM25 retriever is required for BM25 or hybrid retrieval")
        return self.bm25_retriever.search(q,filters=filters,source_types=source_types,top_k=top_k)
    def _dense(self,q,vec,filters,top_k,source_types=None):
        if self.vector_store is None: raise ValueError("Vector store is required for dense or hybrid retrieval")
        if vec is None:
            if self.embedding_provider is None: raise ValueError("Embedding provider is required when dense_vector is not provided")
            vec=self.embedding_provider.embed_query(q)
        return self.vector_store.dense_search(vec,filters=filters,source_types=source_types,top_k=top_k)
    def retrieve(
        self,
        query: str,
        dense_vector: list[float] | None = None,
        filters: dict | None = None,
        top_k: int = 5,
        source_types: list[str] | None = None,
        mode: str = "hybrid_rrf",
        candidate_k: int | None = None,
        dense_k: int | None = None,
        bm25_k: int | None = None,
        rrf_k: int | None = None,
    ):
        """
        Formal retrieval setting:
        - Dense Top20
        - BM25 Top20
        - RRF Top10
        - Reranker Top5

        candidate_k is kept for backward compatibility.
        If dense_k / bm25_k / rrf_k are not explicitly provided:
        - dense_k falls back to candidate_k or 20
        - bm25_k falls back to candidate_k or 20
        - rrf_k falls back to 10
        """
        top_k = int(top_k or 5)

        if candidate_k is None or candidate_k <= 0:
            candidate_k = 20

        dense_k = int(dense_k or getattr(self, "dense_k", None) or candidate_k or 20)
        bm25_k = int(bm25_k or getattr(self, "bm25_k", None) or candidate_k or 20)
        rrf_k = int(rrf_k or getattr(self, "rrf_k_final", None) or 10)

        dense_k = max(dense_k, top_k)
        bm25_k = max(bm25_k, top_k)
        rrf_k = max(rrf_k, top_k)

        # RRF rank constant, not the number of fused candidates.
        rrf_rank_constant = int(getattr(self, "rrf_rank_constant", 60))

        if mode == "bm25_only":
            return self._bm25(query, filters, bm25_k, source_types)[:top_k]

        if mode == "dense_only":
            return self._dense(query, dense_vector, filters, dense_k, source_types)[:top_k]
        bm25_hits = None
        if mode in {"hybrid_rrf", "hybrid_rerank"}:
            bm25_hits = self._bm25(query, filters, bm25_k, source_types)
            if self._is_exact_bm25_hits(bm25_hits):
                if mode == "hybrid_rerank" and getattr(self, "reranker", None) is not None:
                    return self.reranker.rerank(query, bm25_hits, top_k)
                return bm25_hits[:top_k]



        if mode == "hybrid_rrf":
            bm25_hits = bm25_hits if bm25_hits is not None else self._bm25(query, filters, bm25_k, source_types)
            dense_hits = self._dense(query, dense_vector, filters, dense_k, source_types)

            fused = rrf_fusion(
                [bm25_hits, dense_hits],
                k=rrf_rank_constant,
                top_k=rrf_k,
            )
            return fused[:rrf_k]

        if mode == "hybrid_rerank":
            if self.reranker is None:
                raise ValueError("Reranker is required for hybrid_rerank retrieval")

            bm25_hits = bm25_hits if bm25_hits is not None else self._bm25(query, filters, bm25_k, source_types)
            dense_hits = self._dense(query, dense_vector, filters, dense_k, source_types)

            fused = rrf_fusion(
                [bm25_hits, dense_hits],
                k=rrf_rank_constant,
                top_k=rrf_k,
            )[:rrf_k]

            reranked = self.reranker.rerank(query, fused, top_k=top_k)
            return reranked[:top_k]

        raise ValueError(f"Unsupported retrieval mode: {mode}")
