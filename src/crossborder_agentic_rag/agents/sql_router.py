"""SQL routing over DuckDB exact lookup APIs."""
from __future__ import annotations
from typing import Any
from crossborder_agentic_rag.schemas.queries import QueryPlan
from crossborder_agentic_rag.storage.duckdb_store import DuckDBStore

class SQLRouter:
    def __init__(self, duckdb_store: DuckDBStore | None = None): self.duckdb_store = duckdb_store
    def run(self, plan: QueryPlan) -> list[dict[str, Any]]:
        if self.duckdb_store is None: raise ValueError("DuckDBStore is required for SQL route")
        q=plan.query.lower(); f=plan.filters; calls=[]
        if f.get("patent_number") and ("litigation" in q or "case" in q): calls.append(("lookup_litigation_by_patent", f["patent_number"]))
        elif f.get("patent_id") or f.get("patent_number"): calls.append(("lookup_patent_by_id", f.get("patent_id") or f.get("patent_number")))
        if f.get("registration_number"): calls.append(("lookup_trademark_by_registration_number", f["registration_number"]))
        elif f.get("word_mark") and "nice" in q: calls.append(("lookup_trademark_classes_by_word_mark", f["word_mark"]))
        elif f.get("word_mark") and "goods" in q: calls.append(("lookup_trademark_goods_services_by_word_mark", f["word_mark"]))
        elif f.get("word_mark") and not (f.get("patent_number") or f.get("patent_id") or f.get("case_number")): calls.append(("lookup_trademark_by_word_mark", f["word_mark"]))
        if f.get("case_number"):
            if "part" in q: calls.append(("lookup_litigation_parties_by_case", f["case_number"]))
            elif "document" in q or "docket" in q: calls.append(("lookup_litigation_documents_by_case", f["case_number"]))
            elif "patent" in q: calls.append(("lookup_litigation_patents_by_case", f["case_number"]))
            else: calls.append(("lookup_litigation_by_case", f["case_number"]))
        out=[]
        for name,arg in calls:
            for row in getattr(self.duckdb_store,name)(arg):
                d=dict(row); d["_lookup"]=name; out.append(d)
        return out
