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
    risk_terms = [
        "risk",
        "listing risk",
        "compliance risk",
        "can this product be sold",
        "can i sell",
        "should i list",
        "could this listing infringe",
        "ip risk assessment",
        "counterfeit risk",
        "infringement risk",
        "assess ip risks",
    ]
    if _has(lower, *risk_terms):
        sources: list[str] = []
        if _has(lower, "trademark", "mark", "brand", "logo"):
            sources.append("trademark")
        if _has(lower, "patent", "claim", "design", "invention", "utility"):
            sources.append("patent")
        if _has(lower, "case", "litigation", "lawsuit", "sued"):
            sources.append("litigation")
        if not sources:
            sources = ["trademark", "patent", "litigation"]
        return QueryClassification(q, "risk_analysis", "risk_analysis", "multi_source_risk", sources, rationale="explicit risk/listing language")
    if re.search(r"\b(case|docket|parties|documents)\b", lower) and re.search(r"\d+:\d{2}-cv-\d+", lower):
        return QueryClassification(q, "case_lookup", "direct_field_answer", "sql", ["litigation"], rationale="case exact lookup")
    if _has(lower, "nice class", "nice classes", "goods and services", "registration number", "serial number"):
        return QueryClassification(q, "field_lookup", "direct_field_answer", "sql", ["trademark"], rationale="trademark exact field lookup")
    if _has(lower, "summarize litigation history"):
        return QueryClassification(q, "mixed_search", "litigation_summary", "mixed", ["litigation"], rationale="exact entity plus explanation")
    if "patent" in lower and re.search(r"\bus\d+", lower) and _has(lower, "explain", "summarize", "claims"):
        return QueryClassification(q, "mixed_search", "patent_explanation", "mixed", ["patent"], rationale="exact entity plus explanation")
    litigation_terms = ("litigation", "lawsuit", "sued", "docket", "case number", "case numbers", "parties", "asserted patent", "asserted patents", "docket event", "docket events")
    if _has(lower, *litigation_terms):
        return QueryClassification(q, "litigation_summary", "litigation_summary", "hybrid", ["litigation"], rationale="litigation semantic question")

    if "patent" in lower or "claim" in lower or "invention" in lower or "utility" in lower:
        return QueryClassification(q, "patent_explanation", "patent_explanation", "hybrid", ["patent"], rationale="patent semantic question")
    if "trademark" in lower or "mark" in lower or "brand" in lower or "logo" in lower:
        return QueryClassification(q, "trademark_explanation", "trademark_explanation", "hybrid", ["trademark"], rationale="trademark semantic question")
    if "compare" in lower or "difference" in lower:
        return QueryClassification(q, "comparison", "comparison_answer", "hybrid", [], rationale="comparison question")
    return QueryClassification(q, "general_ip_question", "general_answer", "hybrid", [], rationale="general semantic question")
