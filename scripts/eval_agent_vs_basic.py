#!/usr/bin/env python
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path: sys.path.insert(0,str(ROOT/"src"))
if str(ROOT/"scripts") not in sys.path: sys.path.insert(0,str(ROOT/"scripts"))
from argparse import Namespace
from crossborder_agentic_rag.evaluation.dataset import load_eval_dataset
from crossborder_agentic_rag.evaluation import retrieval_metrics as rm
from crossborder_agentic_rag.evaluation import answer_metrics as am
from crossborder_agentic_rag.evaluation import agent_metrics as agm
from crossborder_agentic_rag.evaluation.reporting import write_jsonl,write_json,write_csv,write_markdown_summary,aggregate_metric_rows
from agentic_rag_cli_common import build_runtime

def manifest(evs): return [{"id":f"E{i+1}","chunk_id":c.chunk_id,"doc_id":c.doc_id,"source_type":c.source_type,"source_subtype":c.source_subtype,"title":c.title,"content":c.content[:450]} for i,c in enumerate(evs)]
def det_answer(query,evidence):
    cites=" ".join(f"[E{i+1}]" for i,_ in enumerate(evidence[:3]))
    base=f"Evidence-based summary for '{query}'. {cites}" if evidence else "Insufficient evidence was found."
    return base+" This is not legal advice; consult qualified counsel."
def retrieval_metric_map(hits,ex):
    out={}
    for k in [5,8]:
        out[f"Precision@{k}"]=rm.precision_at_k(hits,ex,k); out[f"Recall@{k}"]=rm.recall_at_k(hits,ex,k); out[f"HitRate@{k}"]=rm.hit_rate_at_k(hits,ex,k); out[f"MRR@{k}"]=rm.mrr_at_k(hits,ex,k); out[f"nDCG@{k}"]=rm.ndcg_at_k(hits,ex,k)
    out["SourceTypeCoverage@8"]=rm.source_type_coverage(hits,ex.expected_source_types,8); return out
def all_metrics(answer, evidence, ex, trace, tool_calls, mode, latency):
    em=manifest(evidence); m=retrieval_metric_map(evidence,ex)
    m.update({"CitationCoverage":am.citation_coverage(answer,em),"ValidCitationRate":am.valid_citation_rate(answer,em),"GroundedCitationRate":am.grounded_citation_rate(answer,em),"FaithfulnessProxy":am.faithfulness_proxy(answer,em),"AnswerRelevanceProxy":am.answer_relevance_proxy(answer,ex),"MissingEvidenceMentioned":1.0 if am.missing_evidence_mentioned(answer,[]) else 0.0,"NoLegalAdviceWarning":1.0 if am.no_legal_advice_warning(answer) else 0.0,"TraceCompleteness":agm.trace_completeness_score(trace,mode),"ToolCallCount":agm.tool_call_count(tool_calls),"FollowupQueryCount":agm.followup_query_count(tool_calls),"UsedFollowupRetrieval":1.0 if agm.used_followup_retrieval(trace,tool_calls) else 0.0,"AgenticProcessValid":1.0 if agm.agentic_process_valid(trace,tool_calls,mode) else 0.0,"LatencyMs":latency,"final_evidence_count":len(evidence)})
    return m

def main(argv=None):
    p=argparse.ArgumentParser(description="Compare basic_rag and agentic RAG pipelines.")
    p.add_argument("--eval-path",default="eval/queries_small.jsonl"); p.add_argument("--chunks-path",default="data/processed/chunks_qa_300k.jsonl"); p.add_argument("--collection-name",default="ip_chunks_qa_300k"); p.add_argument("--use-milvus", action="store_true"); p.add_argument("--pipeline-modes",default="basic_rag,agentic"); p.add_argument("--retrieval-mode",default="hybrid_rerank"); p.add_argument("--reranker-provider",default="lexical"); p.add_argument("--reranker-model"); p.add_argument("--top-k",type=int,default=8); p.add_argument("--candidate-k",type=int,default=50); p.add_argument("--embedding-provider",default="local"); p.add_argument("--embedding-model"); p.add_argument("--use-llm",action="store_true"); p.add_argument("--llm-provider"); p.add_argument("--llm-model"); p.add_argument("--llm-base-url"); p.add_argument("--llm-judge",action="store_true"); p.add_argument("--max-evidence-for-llm",type=int,default=6); p.add_argument("--max-chars-per-evidence",type=int,default=450); p.add_argument("--llm-max-tokens",type=int,default=800); p.add_argument("--temperature",type=float,default=0.0); p.add_argument("--output-dir",default="reports/eval_agent_vs_basic"); p.add_argument("--limit",type=int); p.add_argument("--continue-on-error",action="store_true",default=True); p.add_argument("--source-types",default="trademark,patent,litigation")
    a=p.parse_args(argv); examples=load_eval_dataset(a.eval_path); examples=examples[:a.limit] if a.limit else examples; rows=[]
    for mode in [x.strip() for x in a.pipeline_modes.split(',') if x.strip()]:
      rt_args=Namespace(**{**vars(a), "embedding_provider": ("fake" if a.retrieval_mode == "bm25_only" else a.embedding_provider), "pipeline_mode": mode, "demo": False, "max_iterations": 2})
      runtime=None
      try:
          runtime=build_runtime(rt_args)
      except Exception as exc:
          if not a.continue_on_error: raise
          for ex in examples:
              rows.append({"id":ex.id,"query":ex.query,"pipeline_mode":mode,"error":str(exc),"metrics":{},"latency_ms":0})
          continue
      for ex in examples:
        st=time.perf_counter()
        try:
            result=runtime.run_query(ex.query)
            evidence = result.get("reranked_evidence") or result.get("retrieved_evidence") or []
            answer = result.get("llm_answer") or result.get("deterministic_answer") or ""
            lat=(time.perf_counter()-st)*1000
            # Metrics accept dict-like evidence manifests for answer checks; retrieval metrics are best-effort here.
            metrics={"LatencyMs":lat,"final_evidence_count":result.get("final_evidence_count",0),"ToolCallCount":result.get("tool_call_count",0),"FollowupQueryCount":result.get("followup_query_count",0),"NoLegalAdviceWarning":1.0 if am.no_legal_advice_warning(answer) else 0.0}
            row={"id":ex.id,"query":ex.query,**result,"metrics":metrics,"latency_ms":lat}
            rows.append(row)
        except Exception as exc:
            if not a.continue_on_error: raise
            rows.append({"id":ex.id,"query":ex.query,"pipeline_mode":mode,"error":str(exc),"metrics":{},"latency_ms":(time.perf_counter()-st)*1000})
    out=Path(a.output_dir); write_jsonl(out/"agent_vs_basic_results.jsonl",rows); groups=aggregate_metric_rows(rows,"pipeline_mode")
    comp=[]; b=groups.get("basic_rag",{}); ag=groups.get("agentic",{})
    for metric in sorted(set(b)|set(ag)):
        bv=b.get(metric,{}).get("mean") if isinstance(b.get(metric),dict) else None; av=ag.get(metric,{}).get("mean") if isinstance(ag.get(metric),dict) else None
        if bv is not None or av is not None: comp.append({"metric":metric,"basic_rag":bv,"agentic":av,"delta":(av-bv) if isinstance(av,(int,float)) and isinstance(bv,(int,float)) else None})
    summary={"run_config":vars(a),"groups":groups,"comparison":comp}; write_json(out/"agent_vs_basic_summary.json",summary); write_csv(out/"agent_vs_basic_summary.csv",comp); write_markdown_summary(out/"agent_vs_basic_summary.md",summary)
if __name__=="__main__": main()
