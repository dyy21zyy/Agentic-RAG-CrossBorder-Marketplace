"""Evidence sufficiency evaluation."""
from __future__ import annotations
from dataclasses import dataclass, field
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
from crossborder_agentic_rag.schemas.queries import QueryPlan

@dataclass(slots=True)
class EvidenceEvaluation:
    is_sufficient: bool
    missing_source_types: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    followup_queries: list[str] = field(default_factory=list)

def build_followup_query(original_query: str, missing_source_type: str) -> str:
    return f"Find {missing_source_type} evidence for: {original_query}"

def _has(evidence, st): return any(c.source_type == st for c in evidence)
def _sql_has(sql, needle): return any(needle in str(r.get("_lookup", "")) for r in (sql or []))

def evaluate_evidence(plan: QueryPlan, evidence: list[EvidenceChunk], sql_results: list[dict] | None = None) -> EvidenceEvaluation:
    sql_results = sql_results or []
    missing=[]
    t=plan.expected_answer_type
    if t=="policy_answer" and not _has(evidence,"policy"): missing.append("policy")
    elif t=="patent_explanation" and not (_has(evidence,"patent") or _sql_has(sql_results,"patent")): missing.append("patent")
    elif t=="trademark_explanation" and not (_has(evidence,"trademark") or _sql_has(sql_results,"trademark")): missing.append("trademark")
    elif t=="litigation_summary" and not (_has(evidence,"litigation") or _sql_has(sql_results,"litigation")): missing.append("litigation")
    elif t=="risk_analysis":
        if "policy" in plan.source_types and "policy" in plan.query.lower() and not _has(evidence,"policy"):
            missing.append("policy")
        if not (_has(evidence,"trademark") or _has(evidence,"patent") or _has(evidence,"litigation") or sql_results):
            for st in ["trademark","patent","litigation"]:
                if st in plan.source_types or not plan.source_types: missing.append(st); break
    elif t=="direct_field_answer":
        if not sql_results and not evidence: missing.append(plan.source_types[0] if plan.source_types else "structured")
    sufficient = not missing
    gaps=[f"Missing {m} evidence" for m in missing]
    return EvidenceEvaluation(sufficient, missing, gaps, [build_followup_query(plan.query,m) for m in missing])
