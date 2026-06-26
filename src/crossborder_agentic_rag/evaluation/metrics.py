"""Deterministic evaluation metrics for Stage 7."""
from __future__ import annotations
import math, re, string
from collections import Counter

_PUNCT=str.maketrans({c:" " for c in string.punctuation})
_TOKEN_RE=re.compile(r"\w+", re.UNICODE)

def _check_k(k:int)->None:
    if k<=0: raise ValueError("k must be positive")

def _dedup(ids:list[str])->list[str]:
    seen=set(); out=[]
    for i in ids:
        if i not in seen:
            seen.add(i); out.append(i)
    return out

def recall_at_k(retrieved_ids:list[str], relevant_ids:list[str], k:int)->float:
    _check_k(k); rel=set(relevant_ids)
    if not rel: return 0.0
    return len(set(_dedup(retrieved_ids)[:k]) & rel)/len(rel)

def precision_at_k(retrieved_ids:list[str], relevant_ids:list[str], k:int)->float:
    _check_k(k); rel=set(relevant_ids); got=_dedup(retrieved_ids)[:k]
    if not rel or not got: return 0.0
    return len(set(got)&rel)/len(got)

def hit_rate_at_k(retrieved_ids:list[str], relevant_ids:list[str], k:int)->float:
    _check_k(k); rel=set(relevant_ids)
    if not rel: return 0.0
    return 1.0 if any(x in rel for x in _dedup(retrieved_ids)[:k]) else 0.0

def mrr_at_k(retrieved_ids:list[str], relevant_ids:list[str], k:int)->float:
    _check_k(k); rel=set(relevant_ids)
    if not rel: return 0.0
    for idx, rid in enumerate(_dedup(retrieved_ids)[:k], start=1):
        if rid in rel: return 1.0/idx
    return 0.0

def average_precision_at_k(retrieved_ids:list[str], relevant_ids:list[str], k:int)->float:
    _check_k(k); rel=set(relevant_ids)
    if not rel: return 0.0
    hits=0; total=0.0
    for idx, rid in enumerate(_dedup(retrieved_ids)[:k], start=1):
        if rid in rel:
            hits+=1; total += hits/idx
    return total/min(len(rel), k)

def map_at_k(batch_retrieved_ids:list[list[str]], batch_relevant_ids:list[list[str]], k:int)->float:
    _check_k(k)
    if len(batch_retrieved_ids)!=len(batch_relevant_ids): raise ValueError("batch lengths must match")
    return mean([average_precision_at_k(r, rel, k) for r,rel in zip(batch_retrieved_ids,batch_relevant_ids)])

def ndcg_at_k(retrieved_ids:list[str], relevant_ids:list[str], k:int)->float:
    _check_k(k); rel=set(relevant_ids)
    if not rel: return 0.0
    dcg=0.0
    for idx, rid in enumerate(_dedup(retrieved_ids)[:k], start=1):
        if rid in rel: dcg += 1.0/math.log2(idx+1)
    ideal=min(len(rel), k)
    idcg=sum(1.0/math.log2(i+1) for i in range(1, ideal+1))
    return 0.0 if idcg==0 else dcg/idcg

def _same_len(a,b):
    if len(a)!=len(b): raise ValueError("length mismatch")

def routing_accuracy(predicted_routes:list[str], expected_routes:list[str])->float:
    _same_len(predicted_routes, expected_routes)
    return mean([1.0 if p==e else 0.0 for p,e in zip(predicted_routes, expected_routes)])

def source_type_accuracy_strict(predicted_source_types:list[list[str]], expected_source_types:list[list[str]])->float:
    _same_len(predicted_source_types, expected_source_types)
    return mean([1.0 if set(p)==set(e) else 0.0 for p,e in zip(predicted_source_types, expected_source_types)])

def source_type_accuracy_loose(predicted_source_types:list[list[str]], expected_source_types:list[list[str]])->float:
    _same_len(predicted_source_types, expected_source_types); vals=[]
    for p,e in zip(predicted_source_types, expected_source_types):
        ps, es=set(p), set(e)
        vals.append(1.0 if ((not ps and not es) or bool(ps & es)) else 0.0)
    return mean(vals)

def normalize_answer(text:str)->str:
    return " ".join((text or "").lower().translate(_PUNCT).split())

def exact_match(prediction:str, gold:str)->float:
    return 1.0 if normalize_answer(prediction)==normalize_answer(gold) else 0.0

def token_f1(prediction:str, gold:str)->float:
    pt=normalize_answer(prediction).split(); gt=normalize_answer(gold).split()
    if not pt and not gt: return 1.0
    if not pt or not gt: return 0.0
    common=Counter(pt) & Counter(gt); overlap=sum(common.values())
    if overlap==0: return 0.0
    prec=overlap/len(pt); rec=overlap/len(gt)
    return 2*prec*rec/(prec+rec)

def citation_coverage(answer_citations:list[str], required_relevant_ids:list[str])->float:
    if not required_relevant_ids: return 1.0
    covered=sum(1 for rid in set(required_relevant_ids) if any(rid in c for c in answer_citations))
    return covered/len(set(required_relevant_ids))

def grounded_citation_rate(answer_citations:list[str], retrieved_ids:list[str])->float:
    if not answer_citations: return 0.0
    grounded=sum(1 for c in answer_citations if any(rid and rid in c for rid in retrieved_ids))
    return grounded/len(answer_citations)

def faithfulness_proxy(answer:str, evidence_texts:list[str], citations:list[str])->float:
    if not normalize_answer(answer): return 0.0
    ans=set(normalize_answer(answer).split()); ev=set(normalize_answer(" ".join(evidence_texts)).split())
    overlap=(len(ans & ev)/len(ans)) if ans and ev else 0.0
    cite_bonus=1.0 if citations else (0.4 if evidence_texts else 0.8)
    return max(0.0, min(1.0, 0.75*overlap + 0.25*cite_bonus))

def mean(values:list[float])->float:
    return sum(values)/len(values) if values else 0.0

def safe_round(value:float, digits:int=4)->float:
    try: return round(float(value), digits)
    except (TypeError, ValueError): return 0.0
