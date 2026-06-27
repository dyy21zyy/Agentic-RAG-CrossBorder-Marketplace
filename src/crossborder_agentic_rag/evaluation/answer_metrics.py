"""Heuristic answer and citation metrics for development evaluation."""
from __future__ import annotations
import re
from typing import Any
_CIT_RE=re.compile(r"\[(E\d+)\]")
_TOK_RE=re.compile(r"\w+")
_STOP={"the","a","an","and","or","to","of","in","for","on","with","is","are","be","this","that","as","by","it","not","legal","advice"}
def _get(o,n,d=""): return o.get(n,d) if isinstance(o,dict) else getattr(o,n,d)
def _t(s): return {x.lower() for x in _TOK_RE.findall(s or "") if len(x)>2 and x.lower() not in _STOP}
def extract_citation_ids(answer:str)->list[str]:
    out=[]; seen=set()
    for c in _CIT_RE.findall(answer or ""):
        if c not in seen: seen.add(c); out.append(c)
    return out
def _ids(manifest): return [str(_get(e,"id","") or _get(e,"citation_id","") or f"E{i+1}") for i,e in enumerate(manifest or [])]
def _snip(e): return " ".join(str(_get(e,k,"")) for k in ("title","content","snippet","content_preview"))
def citation_coverage(answer,evidence_manifest):
    ids=set(_ids(evidence_manifest))
    if not ids: return None
    return len(ids & set(extract_citation_ids(answer)))/len(ids)
def valid_citation_rate(answer,evidence_manifest):
    cites=extract_citation_ids(answer)
    if not cites: return 0.0 if (answer or "").strip() else None
    ids=set(_ids(evidence_manifest)); return sum(1 for c in cites if c in ids)/len(cites)
def grounded_citation_rate(answer,evidence_manifest):
    """Proxy: cited sentence is supported when it overlaps cited evidence tokens."""
    by_id={cid:e for cid,e in zip(_ids(evidence_manifest), evidence_manifest or [])}; total=ok=0
    for sent in re.split(r"(?<=[.!?])\s+", answer or ""):
        cites=extract_citation_ids(sent)
        if not cites: continue
        total+=1; st=_t(_CIT_RE.sub("",sent)); ev=set().union(*[_t(_snip(by_id[c])) for c in cites if c in by_id]) if cites else set()
        if st and ev and len(st & ev)/max(1,len(st)) >= 0.2: ok+=1
    return None if total==0 else ok/total
def answer_relevance_proxy(answer, example):
    if not (answer or "").strip(): return None
    label=getattr(example,"gold_answer","") or ""
    if label:
        a,g=_t(answer),_t(label); return len(a & g)/len(g) if g else None
    q=_t(getattr(example,"query","") or ""); must={x.lower() for x in getattr(example,"must_contain_any",[]) or []}; a=_t(answer)
    denom=len(q|must)
    return len(a & (q|must))/denom if denom else None
def faithfulness_proxy(answer,evidence_manifest):
    """Heuristic 0-1 proxy; not a substitute for human/legal review."""
    a=_t(_CIT_RE.sub("",answer or "")); ev=_t(" ".join(_snip(e) for e in evidence_manifest or []))
    if not a: return None
    if not ev: return 0.0
    return max(0.0,min(1.0,len(a & ev)/len(a)))
def missing_evidence_mentioned(answer,evidence_gaps):
    if not evidence_gaps: return False
    text=(answer or "").lower()
    return any(x in text for x in ["missing evidence","insufficient evidence","not enough evidence","evidence is limited","could not find"])
def no_legal_advice_warning(answer):
    text=(answer or "").lower()
    return "not legal advice" in text or "not a substitute for legal advice" in text or "consult" in text and "attorney" in text
