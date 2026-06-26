#!/usr/bin/env python
"""Run Stage 7 ablation experiments."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
from crossborder_agentic_rag.evaluation.datasets import load_eval_jsonl
from crossborder_agentic_rag.evaluation.ablations import default_ablation_configs, run_ablations
from crossborder_agentic_rag.evaluation.report import write_ablation_report_bundle
import importlib.util
spec=importlib.util.spec_from_file_location("eval09", ROOT/"scripts/09_run_eval.py"); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--eval-file",required=True); p.add_argument("--output-dir",default="data/eval/ablation"); p.add_argument("--duckdb-path"); p.add_argument("--chunks-path"); p.add_argument("--use-milvus",action="store_true"); p.add_argument("--collection-name"); p.add_argument("--embedding-provider",default="fake"); p.add_argument("--experiments"); p.add_argument("--demo",action="store_true"); return p.parse_args()
def main():
    args=parse_args(); f=Path(args.eval_file)
    if not f.exists(): print(f"Missing eval file: {f}", file=sys.stderr); return 2
    if not args.demo and not (args.chunks_path or args.use_milvus): print("Normal mode requires --chunks-path or --use-milvus", file=sys.stderr); return 2
    examples=load_eval_jsonl(f); configs=default_ablation_configs()
    if args.experiments:
        wanted=set(args.experiments.split(",")); configs=[c for c in configs if c.name in wanted]
    def factory(cfg): return mod.DemoAgent(source_types=cfg.source_types)
    results=run_ablations(examples, factory, configs=configs, top_ks=[5,10])
    paths=write_ablation_report_bundle(results,args.output_dir)
    for r in results: print(f"{r.name}: Recall@5={r.summary_metrics.get('Recall@5')} MAP@10={r.summary_metrics.get('MAP@10')}")
    print(f"reports={paths}"); return 0
if __name__=="__main__": raise SystemExit(main())
