"""Phase 4 JSONL evaluation dataset schema."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

@dataclass(slots=True)
class EvalExample:
    id: str
    query: str
    query_type: str = ""
    expected_route: str = ""
    expected_source_types: list[str] = field(default_factory=list)
    expected_source_subtypes: list[str] = field(default_factory=list)
    relevant_doc_ids: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    must_contain_any: list[str] = field(default_factory=list)
    gold_answer: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

_FIELDS=set(EvalExample.__dataclass_fields__)
_LISTS={"expected_source_types","expected_source_subtypes","relevant_doc_ids","relevant_chunk_ids","must_contain_any"}

def _list(v: Any, field_name: str, line_no: int) -> list[str]:
    if v is None: return []
    if not isinstance(v, list): raise ValueError(f"Field {field_name} must be a list on line {line_no}")
    return [str(x) for x in v if x is not None]

def _from_dict(data: dict[str, Any], line_no: int) -> EvalExample:
    if not data.get("id") and data.get("query_id"):
        data=dict(data); data["id"]=data["query_id"]
    if not data.get("id"): raise ValueError(f"Missing id on line {line_no}")
    if not data.get("query"): raise ValueError(f"Missing query on line {line_no}")
    meta=dict(data.get("metadata") or {})
    for k,v in data.items():
        if k not in _FIELDS and k != "query_id": meta[k]=v
    kwargs={k:data.get(k) for k in _FIELDS if k!="metadata"}
    for k in _LISTS: kwargs[k]=_list(kwargs.get(k), k, line_no)
    for k,v in list(kwargs.items()):
        if v is None and k not in _LISTS: kwargs[k]=""
    kwargs["metadata"]=meta
    return EvalExample(**kwargs)

def load_eval_dataset(path: str | Path) -> list[EvalExample]:
    p=Path(path); out=[]
    with p.open("r", encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            try: data=json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"Invalid JSON on line {line_no}: {exc.msg}") from exc
            if not isinstance(data, dict): raise ValueError(f"Expected JSON object on line {line_no}")
            out.append(_from_dict(data,line_no))
    return out
