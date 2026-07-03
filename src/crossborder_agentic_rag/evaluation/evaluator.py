"""Evaluation runner."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time
from crossborder_agentic_rag.evaluation.datasets import EvalExample
from crossborder_agentic_rag.evaluation.metrics import *

@dataclass(slots=True)
class EvalResult:
    query_id: str; query: str; predicted_route: str; expected_route: str; predicted_answer_type: str; expected_answer_type: str; answer: str; gold_answer: str
    citations: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_contexts: list[str] = field(default_factory=list)
    predicted_source_types: list[str] = field(default_factory=list)
    expected_source_types: list[str] = field(default_factory=list)
    predicted_tools: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    predicted_partitions: list[str] = field(default_factory=list)
    expected_partitions: list[str] = field(default_factory=list)
    latency_breakdown: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    trace: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class EvalSummary:
    num_examples: int
    metrics: dict[str, float]
    by_query_type: dict[str, dict[str, float]] = field(default_factory=dict)
    by_route: dict[str, dict[str, float]] = field(default_factory=dict)

def _uniq(xs):
    out=[]; seen=set()
    for x in xs:
        if x and x not in seen: seen.add(x); out.append(x)
    return out

def _chunks(state): return list(getattr(state,"reranked_evidence",[]) or getattr(state,"retrieved_evidence",[]) or [])

def _group(results, examples, keyfn):
    groups={}
    for r,e in zip(results, examples): groups.setdefault(keyfn(r,e) or "unknown", []).append(r)
    return {k:_aggregate(v, name_map=False) for k,v in groups.items()}

def _aggregate(results:list[EvalResult], name_map:bool=True)->dict[str,float]:
    keys=sorted({k for r in results for k in r.metrics})
    agg={k:safe_round(mean([r.metrics.get(k,0.0) for r in results])) for k in keys}
    if name_map:
        if "RoutingCorrect" in agg: agg["RoutingAccuracy"]=agg.pop("RoutingCorrect")
        if "SourceTypeStrict" in agg: agg["SourceTypeAccuracyStrict"]=agg.pop("SourceTypeStrict")
        if "SourceTypeLoose" in agg: agg["SourceTypeAccuracyLoose"]=agg.pop("SourceTypeLoose")
        for k in list(agg):
            if k.startswith("AP@"): agg["MAP@"+k.split("@",1)[1]]=agg.pop(k)
        if "LatencyMs" in agg: agg["LatencyMsMean"]=agg.pop("LatencyMs")
    return agg

def _extract_tool_calls(state: Any) -> list[str]:
    calls=[]
    raw=list(getattr(state,"tool_calls",[]) or [])
    meta=getattr(state,"metadata",{}) or {}
    if isinstance(meta, dict): raw += list(meta.get("tool_calls",[]) or meta.get("intermediate_steps",[]) or [])
    raw += list(getattr(state,"intermediate_steps",[]) or [])
    for call in raw:
        name=None
        if isinstance(call, dict): name=call.get("tool") or call.get("name")
        elif isinstance(call, tuple) and call: name=getattr(call[0],"tool",None) or getattr(call[0],"name",None)
        else: name=getattr(call,"tool",None) or getattr(call,"name",None)
        if name: calls.append(str(name))
    return _uniq(calls)

def _extract_partitions(state: Any, chunks: list[Any]) -> list[str]:
    explicit=list(getattr(state,"partitions_used",[]) or [])
    if explicit: return _uniq([str(x) for x in explicit])
    vals=[]
    for c in chunks:
        md=getattr(c,"metadata",{}) or {}
        if isinstance(md, dict) and md.get("partition"): vals.append(str(md["partition"]))
    return _uniq(vals)

def _latency_metrics(breakdown: dict[str,float], latency: float) -> tuple[float,float,float]:
    if not breakdown: return latency, 0.0, 0.0
    total=float(breakdown.get("total_ms", latency) or latency)
    retrieval=sum(float(breakdown.get(k,0.0) or 0.0) for k in ["embedding_ms","milvus_ms","bm25_ms","duckdb_ms","graph_ms","fusion_ms","rerank_ms","parent_context_ms"])
    generation=float(breakdown.get("llm_ms",0.0) or 0.0)
    return total, retrieval, generation

def evaluate_agent(agent: Any, examples: list[EvalExample], top_ks: list[int] | None = None) -> tuple[list[EvalResult], EvalSummary]:
    ks=sorted(set(top_ks or [5,10]) | {5,10}); results=[]
    for ex in examples:
        start=time.perf_counter(); state=agent.run(ex.query); latency=(time.perf_counter()-start)*1000
        chunks=_chunks(state); chunk_ids=[getattr(c,"chunk_id","") for c in chunks]; doc_ids=[getattr(c,"doc_id","") for c in chunks]
        retrieved_contexts=[getattr(c,"content","") for c in chunks if getattr(c,"content","")]
        srcs=_uniq([getattr(c,"source_type","") for c in chunks]); citations=list(getattr(state,"citations",[]) or [])
        answer=getattr(state,"answer","") or ""; route=getattr(state,"retrieval_route","") or ""; ans_type=getattr(state,"expected_answer_type","") or ""
        rel=ex.relevant_chunk_ids or ex.relevant_doc_ids; got=chunk_ids if ex.relevant_chunk_ids else doc_ids
        predicted_tools=_extract_tool_calls(state); predicted_partitions=_extract_partitions(state,chunks)
        breakdown=dict(getattr(state,"latency_breakdown",{}) or {})
        total_ms,retrieval_ms,generation_ms=_latency_metrics(breakdown, latency)
        m={}
        for k in ks:
            m[f"Recall@{k}"]=recall_at_k(got, rel, k); m[f"Precision@{k}"]=precision_at_k(got, rel, k); m[f"HitRate@{k}"]=hit_rate_at_k(got, rel, k)
            m[f"MRR@{k}"]=mrr_at_k(got, rel, k); m[f"AP@{k}"]=average_precision_at_k(got, rel, k); m[f"nDCG@{k}"]=ndcg_at_k(got, rel, k); m[f"graded_nDCG@{k}"]=ndcg_at_k_graded(got, ex.relevance_grades, k)
        req=ex.relevant_chunk_ids or ex.relevant_doc_ids
        m.update({"ExactMatch":exact_match(answer, ex.gold_answer),"TokenF1":token_f1(answer, ex.gold_answer),"CitationCoverage":citation_coverage(citations, req),"GroundedCitationRate":grounded_citation_rate(citations, chunk_ids+doc_ids),"FaithfulnessProxy":faithfulness_proxy(answer, [getattr(c,"content","") for c in chunks], citations),"RoutingCorrect":1.0 if route==ex.expected_route else 0.0,"SourceTypeStrict":source_type_accuracy_strict([srcs],[ex.expected_source_types]),"SourceTypeLoose":source_type_accuracy_loose([srcs],[ex.expected_source_types]),"ToolCallAccuracy":tool_call_accuracy(predicted_tools, ex.expected_tools),"ToolCallF1":tool_call_f1(predicted_tools, ex.expected_tools),"PartitionAccuracy":partition_accuracy(predicted_partitions, ex.expected_partitions),"LatencyMs":latency,"LatencyTotalMs":total_ms,"LatencyRetrievalMs":retrieval_ms,"LatencyGenerationMs":generation_ms})
        results.append(EvalResult(ex.query_id,ex.query,route,ex.expected_route,ans_type,ex.expected_answer_type,answer,ex.gold_answer,citations,chunk_ids,doc_ids,retrieved_contexts,srcs,ex.expected_source_types,predicted_tools,ex.expected_tools,predicted_partitions,ex.expected_partitions,breakdown,m,latency,list(getattr(state,"trace",[]) or []),dict(ex.metadata)))
    summary_metrics=_aggregate(results)
    summary_metrics.update({
        "LatencyP50": safe_round(percentile([r.metrics.get("LatencyMs",0.0) for r in results],50)),
        "LatencyP90": safe_round(percentile([r.metrics.get("LatencyMs",0.0) for r in results],90)),
        "LatencyP95": safe_round(percentile([r.metrics.get("LatencyMs",0.0) for r in results],95)),
        "LatencyTotalMsMean": safe_round(mean([r.metrics.get("LatencyTotalMs",0.0) for r in results])),
        "LatencyRetrievalMsMean": safe_round(mean([r.metrics.get("LatencyRetrievalMs",0.0) for r in results])),
        "LatencyGenerationMsMean": safe_round(mean([r.metrics.get("LatencyGenerationMs",0.0) for r in results])),
    })
    return results, EvalSummary(len(examples), summary_metrics, _group(results,examples,lambda r,e:e.query_type), _group(results,examples,lambda r,e:r.predicted_route))
