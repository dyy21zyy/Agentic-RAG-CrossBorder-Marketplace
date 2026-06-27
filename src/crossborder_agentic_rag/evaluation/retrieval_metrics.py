"""Phase 4 retrieval quality metrics with weak-label support."""
from __future__ import annotations
import math, re, json
from typing import Any
_TOKEN_RE=re.compile(r"\w+", re.UNICODE)

def _get(obj: Any, name: str, default: Any="") -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
def _tokens(s: str) -> set[str]: return set(_TOKEN_RE.findall((s or "").lower()))
def _text(hit: Any) -> str:
    md=_get(hit,"metadata",{}) or {}
    return " ".join([str(_get(hit,"title","")), str(_get(hit,"content","")), json.dumps(md, ensure_ascii=False)])
def _ids(ex: Any) -> tuple[set[str], set[str]]: return set(getattr(ex,"relevant_chunk_ids",[]) or []), set(getattr(ex,"relevant_doc_ids",[]) or [])
def has_relevance_labels(example: Any) -> bool:
    c,d=_ids(example)
    return bool(c or d or (getattr(example,"expected_source_types",[]) and getattr(example,"must_contain_any",[])))
def is_relevant_hit(hit: Any, example: Any) -> bool | None:
    chunks, docs=_ids(example)
    if chunks or docs: return _get(hit,"chunk_id") in chunks or _get(hit,"doc_id") in docs
    exp=set(getattr(example,"expected_source_types",[]) or []); terms=[t.lower() for t in (getattr(example,"must_contain_any",[]) or [])]
    if not exp and not terms: return None
    if exp and _get(hit,"source_type") not in exp: return False
    if terms and not any(t in _text(hit).lower() for t in terms): return False
    return True

def _rels(hits, ex, k):
    if not has_relevance_labels(ex): return None
    vals=[is_relevant_hit(h,ex) for h in list(hits)[:k]]
    return [bool(v) for v in vals if v is not None]
def precision_at_k(hits, example, k:int):
    r=_rels(hits,example,k)
    if r is None: return None
    return sum(r)/len(r) if r else 0.0
def recall_at_k(hits, example, k:int):
    if not has_relevance_labels(example): return None
    chunks,docs=_ids(example)
    if chunks or docs:
        denom=len(chunks or docs); return sum(1 for h in list(hits)[:k] if is_relevant_hit(h,example))/denom if denom else None
    # weak labels: denominator is at least one expected relevant item; cap at 1
    return 1.0 if any(is_relevant_hit(h,example) for h in list(hits)[:k]) else 0.0
def hit_rate_at_k(hits, example, k:int):
    r=_rels(hits,example,k)
    if r is None: return None
    return 1.0 if any(r) else 0.0
def mrr_at_k(hits, example, k:int):
    if not has_relevance_labels(example): return None
    for i,h in enumerate(list(hits)[:k],1):
        if is_relevant_hit(h,example): return 1.0/i
    return 0.0
def ndcg_at_k(hits, example, k:int):
    if not has_relevance_labels(example): return None
    rel=[1.0 if is_relevant_hit(h,example) else 0.0 for h in list(hits)[:k]]
    dcg=sum(v/math.log2(i+2) for i,v in enumerate(rel))
    chunks,docs=_ids(example); ideal_n=int(min(len(chunks or docs) if (chunks or docs) else sum(rel), k))
    if ideal_n<=0: return 0.0
    idcg=sum(1.0/math.log2(i+2) for i in range(ideal_n))
    return 0.0 if idcg==0 else dcg/idcg
def source_type_coverage(hits, expected_source_types, k:int):
    exp=set(expected_source_types or [])
    if not exp: return None
    got={_get(h,"source_type") for h in list(hits)[:k]}
    return len(exp & got)/len(exp)
def source_type_strict_match(hits, expected_source_types, k:int):
    exp=set(expected_source_types or [])
    if not exp: return None
    got={_get(h,"source_type") for h in list(hits)[:k] if _get(h,"source_type")}
    return 1.0 if got == exp else 0.0
def source_subtype_coverage(hits, expected_source_subtypes, k:int):
    exp=set(expected_source_subtypes or [])
    if not exp: return None
    got={_get(h,"source_subtype") for h in list(hits)[:k]}
    return len(exp & got)/len(exp)
