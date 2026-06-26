"""Retrieval planning for Stage 6."""
from __future__ import annotations
import re
from crossborder_agentic_rag.agents.classify import QueryClassification
from crossborder_agentic_rag.schemas.queries import QueryPlan

_STOP = {"WHICH","WHAT","DOES","BELONG","COVER","FIND","SHOW","CASE","PATENT","TEMU","POLICY","NICE","CLASSES","CLASS","GOODS","SERVICES","REGISTRATION","NUMBER","SERIAL","EXPLAIN","SUMMARIZE","LITIGATION","HISTORY","FOR","THE","OF","ABOUT","USING","LOGO","ON","CAN","SELL","A","AN","THIS","PRODUCT","BRANDED","PRODUCTS","IMPLY"}

def extract_basic_filters(query: str, classification: QueryClassification) -> dict:
    q = query or classification.normalized_query
    filters = dict(classification.filters)
    if m := re.search(r"\bregistration\s+number\s+([A-Za-z0-9-]+)", q, re.I): filters["registration_number"] = m.group(1)
    if m := re.search(r"\bserial\s+number\s+([A-Za-z0-9-]+)", q, re.I): filters["serial_number"] = m.group(1)
    if m := re.search(r"\b(US\d+[A-Z0-9]*)\b", q, re.I):
        filters["patent_number"] = m.group(1).upper(); filters.setdefault("patent_id", filters["patent_number"])
    if m := re.search(r"\bcase\s+([0-9]+:[0-9]{2}-cv-[0-9]+)\b", q, re.I): filters["case_number"] = m.group(1)
    elif m := re.search(r"\b([0-9]+:[0-9]{2}-cv-[0-9]+)\b", q, re.I): filters["case_number"] = m.group(1)
    for st in classification.source_types:
        filters.setdefault("source_type", st if len(classification.source_types) == 1 else classification.source_types)
    caps = [t.strip("?.!,;:") for t in q.split() if re.fullmatch(r"[A-Z][A-Z0-9&-]{2,}", t.strip("?.!,;:")) and t.strip("?.!,;:") not in _STOP]
    if caps and "word_mark" not in filters:
        filters["word_mark"] = caps[0]
    return filters

def build_query_plan(query: str, classification: QueryClassification, top_k: int = 20) -> QueryPlan:
    return QueryPlan(query=classification.normalized_query, query_type=classification.query_type, expected_answer_type=classification.expected_answer_type, retrieval_route=classification.retrieval_route, filters=extract_basic_filters(query, classification), source_types=list(classification.source_types), top_k=top_k)
