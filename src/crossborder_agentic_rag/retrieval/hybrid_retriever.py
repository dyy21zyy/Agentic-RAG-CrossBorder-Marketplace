"""Hybrid retrieval orchestration."""
from __future__ import annotations
from typing import Any
from crossborder_agentic_rag.llm.embeddings import BaseEmbeddingProvider
from crossborder_agentic_rag.retrieval.bm25 import LocalBM25Retriever
from crossborder_agentic_rag.retrieval.reranker import BaseReranker
from crossborder_agentic_rag.retrieval.rrf_fusion import rrf_fusion
class HybridRetriever:
    def __init__(self,embedding_provider:BaseEmbeddingProvider|None=None,bm25_retriever:LocalBM25Retriever|None=None,vector_store:Any|None=None,reranker:BaseReranker|None=None,rrf_k:int=60)->None:
        self.embedding_provider=embedding_provider; self.bm25_retriever=bm25_retriever; self.vector_store=vector_store; self.reranker=reranker; self.rrf_k=rrf_k
    def _bm25(self,q,filters,top_k,source_types):
        if self.bm25_retriever is None: raise ValueError("BM25 retriever is required")
        return self.bm25_retriever.search(q,filters=filters,source_types=source_types,top_k=top_k)
    def _dense(self,q,vec,filters,top_k):
        if self.vector_store is None: raise ValueError("Vector store is required")
        if vec is None:
            if self.embedding_provider is None: raise ValueError("Embedding provider is required when dense_vector is not provided")
            vec=self.embedding_provider.embed_query(q)
        return self.vector_store.dense_search(vec,filters=filters,top_k=top_k)
    def retrieve(self,query:str,dense_vector:list[float]|None=None,filters:dict|None=None,top_k:int=20,source_types:list[str]|None=None,mode:str="hybrid_rrf"):
        if top_k<=0: return []
        if mode=="bm25_only": return self._bm25(query,filters,top_k,source_types)
        if mode=="dense_only": return self._dense(query,dense_vector,filters,top_k)
        if mode=="hybrid_rrf":
            return rrf_fusion([self._bm25(query,filters,top_k,source_types), self._dense(query,dense_vector,filters,top_k)],k=self.rrf_k,top_k=top_k)
        if mode=="hybrid_rerank":
            if self.reranker is None: raise ValueError("Reranker is required")
            return self.reranker.rerank(query,self.retrieve(query,dense_vector,filters,top_k,source_types,"hybrid_rrf"),top_k)
        raise ValueError(f"Unknown retrieval mode: {mode}")
