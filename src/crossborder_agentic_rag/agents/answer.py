"""Deterministic adaptive answer synthesis."""
from __future__ import annotations
from typing import Any
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
from crossborder_agentic_rag.schemas.queries import QueryPlan

def _evidence_citation(c: EvidenceChunk) -> str:
    idx = str(c.metadata.get("chunk_index", c.chunk_id.split(":")[-1] if ":" in c.chunk_id else "0"))
    return f"[{c.source_type}:{c.doc_id}:{c.source_subtype}:{idx}] {c.title} — {c.source_subtype}"

def _sql_citation(r: dict[str, Any]) -> str:
    key = r.get("word_mark") or r.get("registration_number") or r.get("patent_id") or r.get("patent_number") or r.get("case_number") or r.get("doc_id") or "result"
    return f"[sql:{r.get('_lookup','lookup')}:{key}]"

def _line(c: EvidenceChunk) -> str:
    return f"- {c.title}: {c.content[:240]}"

def synthesize_answer(plan: QueryPlan, sql_results: list[dict[str, Any]], evidence: list[EvidenceChunk], evidence_gaps: list[str] | None = None) -> tuple[str, list[str]]:
    citations=[]
    for r in sql_results: citations.append(_sql_citation(r))
    for c in evidence: citations.append(_evidence_citation(c))
    citations=list(dict.fromkeys(citations))
    gaps=evidence_gaps or []
    t=plan.expected_answer_type
    if t=="risk_analysis":
        text=" ".join(c.content.lower() for c in evidence); has_policy=any(c.source_type=="policy" for c in evidence); has_other=bool(sql_results) or any(c.source_type in {"trademark","patent","litigation"} for c in evidence)
        level="Insufficient Evidence" if gaps or not (has_policy and has_other) else ("High" if any(w in text for w in ["counterfeit","remove","removal","suspend"]) else "Medium")
        parts=[f"Risk Level: {level}", "Rationale", "- Based on the gathered policy and IP evidence." if citations else "- Evidence is missing.", "Mitigation Suggestions", "- Verify authorization/licensing and avoid using protected marks or patented features without rights.", "Evidence Used", *( _line(c) for c in evidence[:5]), "Citations", *(citations or ["Evidence is missing."])]
        return "\n".join(parts), citations
    headings={"direct_field_answer":"Structured Results","policy_answer":"Relevant Policy Evidence","patent_explanation":"Relevant Patent Evidence","trademark_explanation":"Relevant Trademark Evidence","litigation_summary":"Case / Litigation Evidence","comparison_answer":"Comparison Points","general_answer":"Evidence"}
    parts=["Answer", f"- Query: {plan.query}", headings.get(t,"Evidence")]
    if sql_results:
        parts += [f"- { {k:v for k,v in r.items() if k != 'metadata'} }" for r in sql_results[:5]]
    if evidence:
        parts += [_line(c) for c in evidence[:5]]
    if not sql_results and not evidence: parts.append("- Evidence is missing.")
    if gaps: parts += ["Evidence Gaps", *[f"- {g}" for g in gaps]]
    parts += ["Citations / Sources" if t=="direct_field_answer" else "Citations", *(citations or ["Evidence is missing."])]
    return "\n".join(parts), citations
