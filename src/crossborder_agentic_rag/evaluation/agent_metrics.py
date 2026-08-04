"""Agent trace/tool-call behavior metrics."""
from __future__ import annotations
from typing import Any

from crossborder_agentic_rag.schemas import RiskScreeningReport

RETRIEVAL_STEPS={"hybrid_retrieval","mixed_hybrid_retrieval","multi_source_risk_retrieval","sql_lookup","followup_retrieval"}
def _step(x): return x.get("step") or x.get("name") or x.get("tool") if isinstance(x,dict) else str(x)
def _steps(trace): return [_step(x) for x in (trace or [])]
def trace_contains_steps(trace, required_steps):
    s=set(_steps(trace)); return all(r in s for r in required_steps)
def trace_completeness_score(trace, pipeline_mode):
    s=set(_steps(trace))
    if pipeline_mode=="basic_rag": req=["basic_rag_direct_retrieval","final_answer"]
    else: req=["normalize_query","classify_query","plan_retrieval","evaluate_evidence","final_answer"]
    got=sum(1 for r in req if r in s)
    if pipeline_mode=="agentic": got += 1 if s & RETRIEVAL_STEPS else 0; denom=len(req)+1
    else: denom=len(req)
    return got/denom if denom else 0.0
def followup_query_count(tool_calls):
    n=0
    for c in tool_calls or []:
        p=c.get("payload",c) if isinstance(c,dict) else {}
        if isinstance(p,dict) and p.get("followup_query"): n+=1
    return n
def tool_call_count(tool_calls): return len(tool_calls or [])
def used_followup_retrieval(trace, tool_calls): return "followup_retrieval" in set(_steps(trace)) or followup_query_count(tool_calls)>0
def agentic_process_valid(trace, tool_calls, pipeline_mode="agentic"):
    s=set(_steps(trace))
    if pipeline_mode=="basic_rag": return not ({"classify_query","plan_retrieval"} & s) and "basic_rag_direct_retrieval" in s
    return {"classify_query","plan_retrieval","evaluate_evidence"} <= s


def agent_metrics(report: RiskScreeningReport) -> dict[str, float | int]:
    """Return report-level agent metrics for a fixture evaluation."""
    outcomes = []
    for result in report.module_results:
        if isinstance(result, dict):
            outcomes.append(str(result.get("status") or result.get("tool_status") or "ok"))
    failures = sum(1 for outcome in outcomes if outcome.lower() in {"failed", "error", "tool_error"})
    return {
        "tool_failure_rate": failures / len(outcomes) if outcomes else 0.0,
        "missing_evidence_count": len(report.missing_evidence),
        "evidence_count": len(report.evidence_items),
    }


compute_agent_metrics = agent_metrics
