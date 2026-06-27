"""Reranking implementations."""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
from crossborder_agentic_rag.retrieval.bm25 import tokenize

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self,query:str,candidates:list[EvidenceChunk],top_k:int)->list[EvidenceChunk]: ...

class NoOpReranker(BaseReranker):
    def rerank(self,query:str,candidates:list[EvidenceChunk],top_k:int)->list[EvidenceChunk]: return [] if top_k<=0 else candidates[:top_k]

def _copy_with_metadata(c: EvidenceChunk, score: float, metadata: dict) -> EvidenceChunk:
    merged=dict(c.metadata); merged.update(metadata)
    return EvidenceChunk(c.chunk_id,c.doc_id,c.source_type,c.source_subtype,c.title,c.content,merged,score)

class LexicalReranker(BaseReranker):
    """Dependency-free reranker based on exact query token overlap."""
    def rerank(self,query:str,candidates:list[EvidenceChunk],top_k:int)->list[EvidenceChunk]:
        if top_k<=0: return []
        q_tokens=tokenize(query)
        q=set(q_tokens)
        scored=[]
        for i,c in enumerate(candidates):
            title_tokens=tokenize(c.title); content_tokens=tokenize(c.content)
            exact_overlap=len(q & set(title_tokens+content_tokens))
            title_overlap=len(q & set(title_tokens))
            # Exact overlap is primary; title matches receive a small boost. Candidate
            # order is preserved by the original index when scores tie.
            score=float(exact_overlap)+(0.25*float(title_overlap))
            scored.append((score,i,_copy_with_metadata(c,score,{"reranker_provider":"lexical","reranker_score":score})))
        scored.sort(key=lambda x:(-x[0],x[1])); return [c for _,_,c in scored[:top_k]]

class LocalCrossEncoderReranker(BaseReranker):
    MESSAGE="sentence-transformers and a working torch installation are required for local cross-encoder reranking. Install the reranker extra and verify torch can import on this machine."
    def __init__(self,model_name:str|None=None)->None:
        self.model_name=model_name or os.getenv("RERANKER_MODEL") or "BAAI/bge-reranker-base"
        try:
            from sentence_transformers import CrossEncoder
            self.model=CrossEncoder(self.model_name)
        except Exception as exc:
            raise ImportError(self.MESSAGE) from exc
    def rerank(self,query:str,candidates:list[EvidenceChunk],top_k:int)->list[EvidenceChunk]:
        if top_k<=0 or not candidates: return []
        pairs=[(query,c.title+"\n"+c.content) for c in candidates]
        scores=[float(s) for s in self.model.predict(pairs)]
        out=[(s,i,_copy_with_metadata(c,s,{"reranker_provider":"local_cross_encoder","reranker_model":self.model_name,"reranker_score":s})) for i,(s,c) in enumerate(zip(scores,candidates))]
        out.sort(key=lambda x:(-x[0],x[1])); return [c for _,_,c in out[:top_k]]

class APIRerankerPlaceholder(BaseReranker):
    def __init__(self)->None:
        self.api_key=os.getenv("RERANKER_API_KEY"); self.api_base=os.getenv("RERANKER_API_BASE"); self.model=os.getenv("RERANKER_MODEL")
    def rerank(self,query:str,candidates:list[EvidenceChunk],top_k:int)->list[EvidenceChunk]:
        raise NotImplementedError("API reranker contract is not implemented in Stage 5; no fake API reranking is performed.")

def build_reranker(provider:str|None=None, model_name:str|None=None)->BaseReranker:
    name=(provider if provider is not None else os.getenv("RERANKER_PROVIDER","noop")).strip().lower().replace("_","-")
    if name in {"noop","none"}: return NoOpReranker()
    if name=="lexical": return LexicalReranker()
    if name in {"local","cross-encoder"}: return LocalCrossEncoderReranker(model_name)
    if name=="api": return APIRerankerPlaceholder()
    raise NotImplementedError(f"Reranker provider '{name}' is not supported. Choose noop, none, lexical, local, cross_encoder, or cross-encoder.")
