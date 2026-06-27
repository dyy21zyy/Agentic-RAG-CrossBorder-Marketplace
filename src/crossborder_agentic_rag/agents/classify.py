"""Deterministic query normalization and classification for Stage 6."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class QueryClassification:
    normalized_query: str
    query_type: str
    expected_answer_type: str
    retrieval_route: str
    source_types: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    rationale: str = ""


def normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    normalized = re.sub(r"\s+", " ", query.strip())
    if not normalized:
        raise ValueError("query must be non-empty")
    return normalized


def _has(text: str, *needles: str) -> bool:
    return any(n in text for n in needles)


def classify_query(query: str) -> QueryClassification:
    q = normalize_query(query)
    lower = q.lower()
    risk_terms = ["risk", "listing risk", "compliance risk", "can this product be sold", "can i sell", "should i list", "does this listing violate policy", "could this listing infringe", "ip risk assessment", "counterfeit risk", "infringement risk"]
    if _has(lower, *risk_terms):
        sources = []
        if _has(lower, "marketplace policy", "platform policy", "temu policy", "temu"):
            sources.append("policy")
        if _has(lower, "mercedes", "logo", "brand", "trademark", "mark"):
            sources.append("trademark")
        if _has(lower, "patent", "claim", "invention"):
            sources.append("patent")
        if _has(lower, "case", "litigation", "lawsuit"):
            sources.append("litigation")
        return QueryClassification(q, "risk_analysis", "risk_analysis", "multi_source_risk", sources, rationale="explicit risk/listing language")
    if re.search(r"\b(case|docket|parties|documents)\b", lower) and re.search(r"\d+:\d{2}-cv-\d+", lower):
        return QueryClassification(q, "case_lookup", "direct_field_answer", "sql", ["litigation"], rationale="case exact lookup")
    if _has(lower, "nice class", "nice classes", "goods and services", "registration number", "serial number"):
        return QueryClassification(q, "field_lookup", "direct_field_answer", "sql", ["trademark"], rationale="trademark exact field lookup")
    if _has(lower, "summarize litigation history") or ("patent" in lower and re.search(r"\bus\d+", lower) and _has(lower, "explain", "summarize", "claims")) or ("policy" in lower and _has(lower, "imply", "branded products")):
        answer = "litigation_summary" if "litigation" in lower else ("patent_explanation" if "patent" in lower else "policy_answer")
        return QueryClassification(q, "mixed_search", answer, "mixed", ["patent"] if "patent" in lower else ["policy", "trademark"], rationale="exact entity plus explanation")
    if "policy" in lower or "temu" in lower:
        return QueryClassification(q, "policy_question", "policy_answer", "hybrid", ["policy"], rationale="policy semantic question")
    if "patent" in lower or "claim" in lower:
        return QueryClassification(q, "patent_explanation", "patent_explanation", "hybrid", ["patent"], rationale="patent semantic question")
    if "trademark" in lower or "mark" in lower or "brand" in lower:
        return QueryClassification(q, "trademark_explanation", "trademark_explanation", "hybrid", ["trademark"], rationale="trademark semantic question")
    if "compare" in lower or "difference" in lower:
        return QueryClassification(q, "comparison", "comparison_answer", "hybrid", [], rationale="comparison question")
    return QueryClassification(q, "general_ip_question", "general_answer", "hybrid", [], rationale="general semantic question")
