#!/usr/bin/env python
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path: sys.path.insert(0,str(ROOT/"src"))
from crossborder_agentic_rag.evaluation.reporting import write_json,write_csv

def flatten(label, obj):
    rows=[]
    if obj.get("comparison"):
        for r in obj["comparison"]: rows.append({"run":label, **r})
    for group, metrics in (obj.get("groups") or {}).items():
        for m,v in metrics.items():
            if isinstance(v,dict): rows.append({"run":label,"group":group,"metric":m,"mean":v.get("mean"),"valid_count":v.get("valid_count")})
    return rows

def main(argv=None):
    p=argparse.ArgumentParser(description="Compare evaluation summary JSON files.")
    p.add_argument("--inputs",nargs="+",required=True); p.add_argument("--labels"); p.add_argument("--output-dir",default="reports/comparisons")
    a=p.parse_args(argv); labels=a.labels.split(',') if a.labels else [Path(x).stem for x in a.inputs]
    if len(labels)!=len(a.inputs): p.error("--labels count must match --inputs")
    rows=[]
    for label,path in zip(labels,a.inputs): rows += flatten(label,json.loads(Path(path).read_text(encoding="utf-8")))
    out=Path(a.output_dir); write_json(out/"comparison.json",{"inputs":a.inputs,"labels":labels,"rows":rows}); write_csv(out/"comparison.csv",rows)
    lines=["# Evaluation Run Comparison","","| Run | Group | Metric | Mean | basic_rag | agentic | delta |","|---|---|---|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r.get('run','')} | {r.get('group','')} | {r.get('metric','')} | {r.get('mean','')} | {r.get('basic_rag','')} | {r.get('agentic','')} | {r.get('delta','')} |")
    (out/"comparison.md").parent.mkdir(parents=True,exist_ok=True); (out/"comparison.md").write_text("\n".join(lines),encoding="utf-8")
if __name__=="__main__": main()
