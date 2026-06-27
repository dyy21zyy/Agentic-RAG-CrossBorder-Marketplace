from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from crossborder_agentic_rag.evaluation.dataset import load_eval_dataset
from crossborder_agentic_rag.evaluation import retrieval_metrics as rm, answer_metrics as am, agent_metrics as agm
from crossborder_agentic_rag.evaluation.reporting import write_jsonl, write_csv, write_markdown_summary
from crossborder_agentic_rag.evaluation.llm_judge import parse_judge_response, run_llm_judge
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk

def c(cid="c1",doc="d1",st="patent",sub="patent_claim",text="drone delivery claim"):
    return EvidenceChunk(cid,doc,st,sub,"Title "+text,text,{})
def test_phase4_dataset_loader_defaults_and_errors(tmp_path):
    p=tmp_path/"q.jsonl"; p.write_text('{"id":"q1","query":"x"}\n',encoding="utf-8")
    ex=load_eval_dataset(p)[0]; assert ex.expected_source_types==[] and ex.must_contain_any==[]
    p.write_text('{"id":"q1"}\n',encoding="utf-8")
    with pytest.raises(ValueError, match="Missing query.*line 1"): load_eval_dataset(p)
    p.write_text('{bad}\n',encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"): load_eval_dataset(p)
def test_retrieval_metrics_strong_weak_missing():
    ex=type("E",(),{"relevant_chunk_ids":["c1"],"relevant_doc_ids":[],"expected_source_types":[],"must_contain_any":[],"expected_source_subtypes":[]})()
    hits=[c("c1"),c("c2")]
    assert rm.precision_at_k(hits,ex,1)==1 and rm.recall_at_k(hits,ex,5)==1 and rm.mrr_at_k(hits,ex,5)==1 and rm.ndcg_at_k(hits,ex,5)>0
    ex2=type("E",(),{"relevant_chunk_ids":[],"relevant_doc_ids":[],"expected_source_types":["patent"],"must_contain_any":["drone"],"expected_source_subtypes":["patent_claim"]})()
    assert rm.hit_rate_at_k(hits,ex2,5)==1 and rm.source_type_coverage(hits,["patent"],5)==1 and rm.source_subtype_coverage(hits,["patent_claim"],5)==1
    ex3=type("E",(),{"relevant_chunk_ids":[],"relevant_doc_ids":[],"expected_source_types":[],"must_contain_any":[]})()
    assert rm.precision_at_k(hits,ex3,5) is None
    assert rm.ndcg_at_k([c("x","x","trademark","trademark_record","zzz")],ex2,5)==0

def test_answer_and_agent_metrics():
    ans="Drone delivery is claimed [E1]. Missing evidence is noted. This is not legal advice. [E9] [E1]"
    ev=[{"id":"E1","title":"Drone delivery","content":"drone delivery claimed"}]
    assert am.extract_citation_ids(ans)==["E1","E9"]
    assert am.valid_citation_rate(ans,ev)==0.5 and am.citation_coverage(ans,ev)==1
    assert am.grounded_citation_rate(ans,ev) is not None
    ex=type("E",(),{"gold_answer":"drone delivery claim","query":"drone delivery","must_contain_any":[]})()
    assert am.answer_relevance_proxy(ans,ex)>0 and am.faithfulness_proxy(ans,ev)>0
    assert am.missing_evidence_mentioned(ans,["litigation"]) and am.no_legal_advice_warning(ans)
    tr=["normalize_query","classify_query","plan_retrieval","hybrid_retrieval","evaluate_evidence","final_answer"]
    assert agm.agentic_process_valid(tr,[],"agentic") and agm.trace_completeness_score(tr,"agentic")==1
    assert agm.agentic_process_valid(["basic_rag_direct_retrieval","final_answer"],[],"basic_rag")
    assert not agm.agentic_process_valid(["basic_rag_direct_retrieval","classify_query","final_answer"],[],"basic_rag")
    assert agm.followup_query_count([{"payload":{"followup_query":"x"}}])==1

def test_reporting_and_llm_judge(tmp_path):
    write_jsonl(tmp_path/"a.jsonl",[{"x":1}]); write_csv(tmp_path/"a.csv",[]); write_markdown_summary(tmp_path/"a.md",{"run_config":{},"groups":{}})
    assert (tmp_path/"a.jsonl").exists() and (tmp_path/"a.csv").exists() and (tmp_path/"a.md").exists()
    good='{"faithfulness_score":1,"answer_relevance_score":1,"citation_correctness_score":1,"completeness_score":1,"rationale":"ok"}'
    assert parse_judge_response(good)["faithfulness_score"]==1
    assert "judge_error" in parse_judge_response("nope")
    assert run_llm_judge(enabled=False)["judge_skipped"]

def test_phase4_scripts_help_and_fixture_runs(tmp_path):
    for s in ["scripts/eval_retrieval.py","scripts/eval_agent_vs_basic.py","scripts/compare_eval_runs.py"]:
        cp=subprocess.run([sys.executable,s,"--help"],cwd=ROOT,text=True,capture_output=True); assert cp.returncode==0, cp.stderr
    chunks=tmp_path/"chunks.jsonl"
    rows=[c("p1","pd1","patent","patent_claim","drone delivery claim").to_dict(), c("t1","td1","trademark","trademark_record","smart bag trademark").to_dict(), c("l1","ld1","litigation","litigation_case","marketplace patent litigation").to_dict()]
    chunks.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    ev=tmp_path/"eval.jsonl"; ev.write_text('{"id":"q1","query":"drone delivery claim","expected_source_types":["patent"],"must_contain_any":["drone"]}\n',encoding="utf-8")
    rdir=tmp_path/"ret"; cp=subprocess.run([sys.executable,"scripts/eval_retrieval.py","--eval-path",str(ev),"--chunks-path",str(chunks),"--modes","bm25_only","--output-dir",str(rdir)],cwd=ROOT,text=True,capture_output=True); assert cp.returncode==0, cp.stderr
    assert (rdir/"retrieval_results.jsonl").exists() and (rdir/"retrieval_summary.md").exists()
    adir=tmp_path/"avb"; cp=subprocess.run([sys.executable,"scripts/eval_agent_vs_basic.py","--eval-path",str(ev),"--chunks-path",str(chunks),"--retrieval-mode","bm25_only","--pipeline-modes","basic_rag,agentic","--output-dir",str(adir)],cwd=ROOT,text=True,capture_output=True); assert cp.returncode==0, cp.stderr
    recs=[json.loads(x) for x in (adir/"agent_vs_basic_results.jsonl").read_text().splitlines()]
    assert "classify_query" not in recs[0]["trace"] and any("classify_query" in r.get("trace",[]) for r in recs)
    assert any("delta" in r for r in json.loads((adir/"agent_vs_basic_summary.json").read_text())["comparison"])
    cdir=tmp_path/"cmp"; cp=subprocess.run([sys.executable,"scripts/compare_eval_runs.py","--inputs",str(adir/"agent_vs_basic_summary.json"),str(adir/"agent_vs_basic_summary.json"),"--labels","a,b","--output-dir",str(cdir)],cwd=ROOT,text=True,capture_output=True); assert cp.returncode==0, cp.stderr
    assert (cdir/"comparison.csv").exists() and (cdir/"comparison.md").exists()
