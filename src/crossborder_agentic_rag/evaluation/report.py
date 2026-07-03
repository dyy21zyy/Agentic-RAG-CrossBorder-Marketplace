"""Evaluation and ablation report writers."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import csv, json
from crossborder_agentic_rag.evaluation.evaluator import EvalResult, EvalSummary
from crossborder_agentic_rag.evaluation.ablations import AblationResult

def _write_json(obj,path):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
def write_eval_results_json(results:list[EvalResult], path:str|Path)->None: _write_json([asdict(r) for r in results], path)
def write_eval_summary_json(summary:EvalSummary, path:str|Path)->None: _write_json(asdict(summary), path)
def write_eval_results_csv(results:list[EvalResult], path:str|Path)->None:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fields=["query_id","query","predicted_route","expected_route","predicted_answer_type","expected_answer_type","answer","gold_answer","citations","retrieved_chunk_ids","retrieved_doc_ids","predicted_source_types","expected_source_types","predicted_tools","expected_tools","predicted_partitions","expected_partitions"]
    metric_keys=sorted({k for r in results for k in r.metrics})
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields+metric_keys); w.writeheader()
        for r in results:
            row={k:getattr(r,k) for k in fields}; row.update(r.metrics); w.writerow(row)
def _md_table(metrics):
    lines=["| Metric | Value |","|---|---:|"]
    for k,v in sorted(metrics.items()): lines.append(f"| {k} | {v} |")
    return "\n".join(lines)
def write_eval_summary_markdown(summary:EvalSummary, path:str|Path)->None:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    s=["# Evaluation Summary","","## Overview",f"- Examples: {summary.num_examples}","","## Main Metrics",_md_table(summary.metrics),"","## By Query Type"]
    for k,m in summary.by_query_type.items(): s += [f"### {k}", _md_table(m)]
    s += ["","## By Route"]
    for k,m in summary.by_route.items(): s += [f"### {k}", _md_table(m)]
    s += ["","## Limitations","Metrics are deterministic proxies; FaithfulnessProxy is heuristic and not a human factuality judge."]
    p.write_text("\n".join(s),encoding="utf-8")
def write_eval_report_bundle(results, summary, output_dir):
    d=Path(output_dir); d.mkdir(parents=True, exist_ok=True); paths={"results_json":d/"eval_results.json","results_csv":d/"eval_results.csv","summary_json":d/"eval_summary.json","summary_md":d/"eval_summary.md"}
    write_eval_results_json(results,paths["results_json"]); write_eval_results_csv(results,paths["results_csv"]); write_eval_summary_json(summary,paths["summary_json"]); write_eval_summary_markdown(summary,paths["summary_md"])
    return {k:str(v) for k,v in paths.items()}
def write_ablation_results_json(results:list[AblationResult], path:str|Path)->None: _write_json([asdict(r) for r in results], path)
def write_ablation_results_csv(results:list[AblationResult], path:str|Path)->None:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True); keys=sorted({k for r in results for k in r.summary_metrics})
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["name","num_examples","config"]+keys); w.writeheader()
        for r in results:
            row={"name":r.name,"num_examples":r.num_examples,"config":json.dumps(r.config,ensure_ascii=False)}; row.update(r.summary_metrics); w.writerow(row)
def write_ablation_summary_markdown(results:list[AblationResult], path:str|Path)->None:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True); base=results[0].summary_metrics if results else {}
    lines=["# Ablation Summary","","| Experiment | Recall@5 | MAP@10 | RoutingAccuracy | Δ Recall@5 |","|---|---:|---:|---:|---:|"]
    b=base.get("Recall@5",0.0)
    for r in results: lines.append(f"| {r.name} | {r.summary_metrics.get('Recall@5','')} | {r.summary_metrics.get('MAP@10','')} | {r.summary_metrics.get('RoutingAccuracy','')} | {round(r.summary_metrics.get('Recall@5',0.0)-b,4)} |")
    lines += ["","## Notes / limitations","Ablation comparison depends on available local/demo backends; failed experiments are marked in metrics/config."]
    p.write_text("\n".join(lines),encoding="utf-8")
def write_ablation_report_bundle(results, output_dir):
    d=Path(output_dir); d.mkdir(parents=True, exist_ok=True); paths={"results_json":d/"ablation_results.json","results_csv":d/"ablation_results.csv","summary_md":d/"ablation_summary.md"}
    write_ablation_results_json(results,paths["results_json"]); write_ablation_results_csv(results,paths["results_csv"]); write_ablation_summary_markdown(results,paths["summary_md"])
    return {k:str(v) for k,v in paths.items()}
