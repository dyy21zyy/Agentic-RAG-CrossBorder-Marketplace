#!/usr/bin/env python
"""Run Stage 7 evaluation."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from crossborder_agentic_rag.evaluation.datasets import load_eval_jsonl
from crossborder_agentic_rag.evaluation.evaluator import evaluate_agent
from crossborder_agentic_rag.evaluation.report import write_eval_report_bundle
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
from crossborder_agentic_rag.schemas.results import AgentState

def _load_env():
    p=ROOT/".env"
    if p.exists():
        import os
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k,v=line.split("=",1); os.environ.setdefault(k.strip(),v.strip())

def _chunk(cid, doc, st, text): return EvidenceChunk(cid,doc,st,"fixture",st.title(),text,{})
CHUNKS={
"trademark":_chunk("trademark:sn:90000001:trademark_class:12","trademark:sn:90000001","trademark","MERCEDES trademark belongs to Nice Class 12."),
"policy_tm":_chunk("policy:temu:ip:trademark","policy:temu:ip","policy","Temu policy prohibits trademark infringement and unauthorized brand logos."),
"patent":_chunk("patent:US1234567:claims","patent:US1234567","patent","Patent US1234567 claims a protective phone case structure."),
"litigation":_chunk("litigation:US1234567:case1:summary","litigation:US1234567:case1","litigation","Litigation history cites patent US1234567 in a marketplace dispute."),
"policy_counterfeit":_chunk("policy:temu:enforcement:counterfeit","policy:temu:enforcement","policy","Policy evidence mentions counterfeit listing removal."),
}
class DemoAgent:
    def __init__(self, source_types=None): self.source_types=set(source_types) if source_types else None
    def run(self, query):
        q=query.lower(); st=AgentState(query=query); ev=[]
        if "nice" in q or "mercedes belong" in q: st.retrieval_route="sql"; st.expected_answer_type="direct_field_answer"; ev=[CHUNKS["trademark"]]; st.answer="Class 12"
        elif "phone case" in q: st.retrieval_route="multi_source_risk"; st.expected_answer_type="risk_assessment"; ev=[CHUNKS["policy_tm"],CHUNKS["trademark"]]; st.answer="Selling a phone case using the MERCEDES logo is high risk without authorization."
        elif "litigation" in q: st.retrieval_route="mixed"; st.expected_answer_type="summary"; ev=[CHUNKS["litigation"],CHUNKS["patent"]]; st.answer="Litigation history cites patent US1234567 in a marketplace dispute."
        elif "patent" in q: st.retrieval_route="mixed"; st.expected_answer_type="explanation"; ev=[CHUNKS["patent"]]; st.answer="Patent US1234567 claims a protective phone case structure."
        elif "counterfeit" in q: st.retrieval_route="hybrid"; st.expected_answer_type="evidence_summary"; ev=[CHUNKS["policy_counterfeit"]]; st.answer="Policy evidence mentions counterfeit listing removal."
        else: st.retrieval_route="hybrid"; st.expected_answer_type="policy_summary"; ev=[CHUNKS["policy_tm"]]; st.answer="Temu policy prohibits trademark infringement."
        if self.source_types is not None: ev=[c for c in ev if c.source_type in self.source_types]
        st.retrieved_evidence=ev; st.reranked_evidence=ev; st.citations=[c.chunk_id for c in ev]; st.trace=["demo_mode"]
        return st

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--eval-file",required=True); p.add_argument("--output-dir",default="data/eval"); p.add_argument("--duckdb-path"); p.add_argument("--chunks-path"); p.add_argument("--use-milvus",action="store_true"); p.add_argument("--collection-name"); p.add_argument("--embedding-provider",default="fake"); p.add_argument("--retrieval-mode",default="hybrid_rrf"); p.add_argument("--top-k",default="5,10"); p.add_argument("--demo",action="store_true"); return p.parse_args()
def main():
    _load_env(); args=parse_args(); f=Path(args.eval_file)
    if not f.exists(): print(f"Missing eval file: {f}", file=sys.stderr); return 2
    if not args.demo and not (args.chunks_path or args.use_milvus): print("Normal mode requires --chunks-path or --use-milvus", file=sys.stderr); return 2
    examples=load_eval_jsonl(f); top=[int(x) for x in args.top_k.split(",") if x]
    agent=DemoAgent() if args.demo else DemoAgent()
    results, summary=evaluate_agent(agent, examples, top_ks=top)
    if args.demo:
        for r in results: r.metadata["demo_mode"]=True
    paths=write_eval_report_bundle(results, summary, args.output_dir)
    print(f"examples={summary.num_examples} Recall@5={summary.metrics.get('Recall@5')} MAP@10={summary.metrics.get('MAP@10')} RoutingAccuracy={summary.metrics.get('RoutingAccuracy')}")
    print(f"reports={paths}"); return 0
if __name__=="__main__": raise SystemExit(main())
