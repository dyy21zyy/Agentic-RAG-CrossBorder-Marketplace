"""Local BM25 retrieval over EvidenceChunk objects."""
from __future__ import annotations
import math, re, copy
from collections import Counter, defaultdict
from typing import Any
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
TOKEN_RE=re.compile(r"[\w]+", re.UNICODE)
def tokenize(text:str)->list[str]: return TOKEN_RE.findall((text or "").lower())
def _copy(c:EvidenceChunk,score:float)->EvidenceChunk:
    return EvidenceChunk(c.chunk_id,c.doc_id,c.source_type,c.source_subtype,c.title,c.content,dict(c.metadata),score)
def _matches(c:EvidenceChunk,filters:dict|None,source_types:list[str]|None)->bool:
    if source_types and c.source_type not in source_types: return False
    for k,v in (filters or {}).items():
        actual=getattr(c,k,None) if hasattr(c,k) else c.metadata.get(k,"")
        if isinstance(v,str):
            if actual!=v: return False
        elif isinstance(v,list):
            if actual not in v: return False
        else: raise ValueError(f"Unsupported filter value for {k}: {v!r}")
    return True
class LocalBM25Retriever:
    def __init__(self,chunks:list[EvidenceChunk],k1:float=1.5,b:float=0.75):
        self.chunks=list(chunks); self.k1=k1; self.b=b; self.docs=[]; self.term_freqs=[]; self.df=defaultdict(int)
        for c in self.chunks:
            text=" ".join([c.title,c.content,c.source_type,c.source_subtype," ".join(str(v) for v in c.metadata.values())])
            toks=tokenize(text); self.docs.append(toks); tf=Counter(toks); self.term_freqs.append(tf)
            for t in tf: self.df[t]+=1
        self.avgdl=(sum(len(d) for d in self.docs)/len(self.docs)) if self.docs else 0.0
    def search(self,query:str,filters:dict|None=None,source_types:list[str]|None=None,top_k:int=20)->list[EvidenceChunk]:
        if top_k<=0 or not self.chunks: return []
        q=tokenize(query)
        if not q: return []
        N=len(self.chunks); out=[]
        for c,toks,tf in zip(self.chunks,self.docs,self.term_freqs):
            if not _matches(c,filters,source_types): continue
            score=0.0; dl=len(toks) or 1
            for term in q:
                if term not in tf: continue
                idf=math.log(1+(N-self.df[term]+0.5)/(self.df[term]+0.5))
                freq=tf[term]; denom=freq+self.k1*(1-self.b+self.b*dl/(self.avgdl or 1))
                score += idf*(freq*(self.k1+1)/denom)
            if score>0: out.append(_copy(c,score))
        return sorted(out,key=lambda x:x.score,reverse=True)[:top_k]
