"""Standard-library report writers for Phase 4 evaluation runs."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import csv,json,statistics

def write_jsonl(path, records):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
def write_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
def write_csv(path, rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); rows=list(rows)
    keys=sorted({k for r in rows for k in r}) or ["empty"]
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
def aggregate_metric_rows(records, group_key):
    groups={}
    for r in records: groups.setdefault(r.get(group_key,"all"),[]).append(r)
    out={}
    for g,rs in groups.items():
        metrics={k for r in rs for k in (r.get("metrics") or {})}; out[g]={}
        for m in metrics:
            vals=[r["metrics"].get(m) for r in rs if (r.get("metrics") or {}).get(m) is not None]
            out[g][m]={"mean":sum(vals)/len(vals) if vals else None,"valid_count":len(vals)}
        lats=[r.get("latency_ms") for r in rs if isinstance(r.get("latency_ms"),(int,float))]
        out[g]["latency_ms"]={"mean":sum(lats)/len(lats) if lats else None,"median":statistics.median(lats) if lats else None,"p95":statistics.quantiles(lats,n=20)[18] if len(lats)>=20 else None,"valid_count":len(lats)}
    return out
def write_markdown_summary(path, summary):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    lines=["# Evaluation Summary","","## Run config","```json",json.dumps(summary.get("run_config",{}),indent=2,ensure_ascii=False),"```",""]
    if "comparison" in summary:
        lines += ["## Agentic vs basic comparison","","| Metric | basic_rag | agentic | delta |","|---|---:|---:|---:|"]
        for row in summary["comparison"]: lines.append(f"| {row.get('metric')} | {row.get('basic_rag','')} | {row.get('agentic','')} | {row.get('delta','')} |")
    lines += ["","## Metrics by group"]
    for group,metrics in (summary.get("groups") or {}).items():
        lines += [f"### {group}","| Metric | Mean | Valid count |","|---|---:|---:|"]
        for m,v in sorted(metrics.items()):
            if isinstance(v,dict): lines.append(f"| {m} | {v.get('mean','')} | {v.get('valid_count','')} |")
    p.write_text("\n".join(lines),encoding="utf-8")
