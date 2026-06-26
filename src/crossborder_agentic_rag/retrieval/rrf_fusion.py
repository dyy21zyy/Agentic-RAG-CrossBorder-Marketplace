"""Reciprocal rank fusion utilities."""
from __future__ import annotations
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
def _copy(c,score): return EvidenceChunk(c.chunk_id,c.doc_id,c.source_type,c.source_subtype,c.title,c.content,dict(c.metadata),score)
def rrf_fusion(result_lists:list[list[EvidenceChunk]],k:int=60,top_k:int|None=None)->list[EvidenceChunk]:
    if k<=0: raise ValueError("k must be positive")
    scores={}; first={}; exemplars={}; order=0
    for lst in result_lists:
        for rank,c in enumerate(lst,1):
            if c.chunk_id not in first:
                first[c.chunk_id]=order; exemplars[c.chunk_id]=c; order+=1
            scores[c.chunk_id]=scores.get(c.chunk_id,0.0)+1.0/(k+rank)
    fused=[_copy(exemplars[cid],score) for cid,score in scores.items()]
    fused.sort(key=lambda c:(-c.score, first[c.chunk_id]))
    return fused if top_k is None else fused[:top_k]
