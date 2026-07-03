"""Evaluation dataset loaders."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

@dataclass(slots=True)
class EvalExample:
    query_id: str
    query: str
    query_type: str = ""
    expected_route: str = ""
    expected_answer_type: str = ""
    gold_answer: str = ""
    relevant_doc_ids: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    expected_source_types: list[str] = field(default_factory=list)
    relevance_grades: dict[str, int] = field(default_factory=dict)
    expected_tools: list[str] = field(default_factory=list)
    expected_partitions: list[str] = field(default_factory=list)
    reference_contexts: list[str] = field(default_factory=list)
    answer_key_points: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

_FIELDS=set(EvalExample.__dataclass_fields__)
_LISTS={"relevant_doc_ids","relevant_chunk_ids","expected_source_types","expected_tools","expected_partitions","reference_contexts","answer_key_points"}

def _example_from_dict(data:dict[str,Any], line_no:int)->EvalExample:
    if not data.get("query_id"): raise ValueError(f"Missing query_id on line {line_no}")
    if not data.get("query"): raise ValueError(f"Missing query on line {line_no}")
    meta=dict(data.get("metadata") or {})
    for k,v in data.items():
        if k not in _FIELDS: meta[k]=v
    kwargs={k:data.get(k) for k in _FIELDS if k not in {"metadata"}}
    kwargs["relevance_grades"] = dict(kwargs.get("relevance_grades") or {})
    for k in _LISTS:
        kwargs[k]=list(kwargs.get(k) or [])
    for k,v in list(kwargs.items()):
        if v is None and k not in _LISTS: kwargs[k]=""
    kwargs["metadata"]=meta
    return EvalExample(**kwargs)

def load_eval_jsonl(path: str | Path) -> list[EvalExample]:
    p=Path(path); out=[]
    with p.open("r", encoding="utf-8") as f:
        for line_no,line in enumerate(f, start=1):
            if not line.strip(): continue
            try: data=json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"Invalid JSON on line {line_no}: {exc.msg}") from exc
            if not isinstance(data, dict): raise ValueError(f"Expected JSON object on line {line_no}")
            out.append(_example_from_dict(data,line_no))
    return out

def write_eval_jsonl(examples: list[EvalExample], path: str | Path) -> int:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False)+"\n")
    return len(examples)
