from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import pytest
from crossborder_agentic_rag.evaluation.metrics import *
from crossborder_agentic_rag.evaluation.datasets import load_eval_jsonl, EvalExample
from crossborder_agentic_rag.evaluation.evaluator import evaluate_agent
from crossborder_agentic_rag.evaluation.ablations import default_ablation_configs, run_ablations
from crossborder_agentic_rag.evaluation.report import write_eval_report_bundle, write_ablation_report_bundle
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
from crossborder_agentic_rag.schemas.results import AgentState

ROOT=Path(__file__).resolve().parents[1]

def test_recall_precision_hit_mrr_ap_map_ndcg():
    r=["a","b","c"]; rel=["b","c","d"]
    assert recall_at_k(r,rel,2)==pytest.approx(1/3)
    assert precision_at_k(r,rel,2)==pytest.approx(1/2)
    assert hit_rate_at_k(r,rel,2)==1.0
    assert mrr_at_k(r,rel,3)==pytest.approx(1/2)
    assert average_precision_at_k(r,rel,3)==pytest.approx(((1/2)+(2/3))/3)
    assert map_at_k([r],[rel],3)==pytest.approx(average_precision_at_k(r,rel,3))
    assert 0 < ndcg_at_k(r,rel,3) <= 1

def test_metric_deduplicates_retrieved_ids():
    assert precision_at_k(["a","a","b"],["a","b"],3)==1.0
    assert recall_at_k(["a","a"],["a"],2)==1.0

def test_metric_edge_cases_empty_relevant_empty_retrieved():
    for fn in [recall_at_k, precision_at_k, hit_rate_at_k, mrr_at_k, average_precision_at_k, ndcg_at_k]: assert fn([],[],5)==0.0
    assert token_f1("","")==1.0

def test_metric_rejects_invalid_k():
    with pytest.raises(ValueError): recall_at_k([],[],0)

def test_routing_accuracy(): assert routing_accuracy(["a","b"],["a","c"])==0.5

def test_source_type_accuracy_strict_and_loose():
    assert source_type_accuracy_strict([["a"],["a","b"],[ ]], [["a"],["b","a"],[]])==1.0
    assert source_type_accuracy_loose([["a"],["c"],[]], [["a","b"],["b"],[]])==pytest.approx(2/3)

def test_source_type_accuracy_length_mismatch():
    with pytest.raises(ValueError): routing_accuracy(["a"],[])

def test_exact_match_case_and_punctuation_insensitive(): assert exact_match("Class 12!","class 12")==1.0

def test_token_f1(): assert token_f1("a b c","a c d")==pytest.approx(2/3)

def test_citation_coverage(): assert citation_coverage(["see chunk-a"],["chunk-a","chunk-b"])==0.5

def test_grounded_citation_rate(): assert grounded_citation_rate(["x id1","bad"],["id1"])==0.5

def test_faithfulness_proxy_range(): assert 0 <= faithfulness_proxy("answer text", ["answer evidence"], ["c1"]) <= 1

def test_eval_dataset_loader_valid_jsonl():
    ex=load_eval_jsonl(ROOT/"tests/fixtures/eval/eval_queries.jsonl")
    assert len(ex)>=6 and ex[0].query_id=="q001" and ex[0].relevant_chunk_ids

def test_eval_dataset_loader_rejects_missing_query_id(tmp_path):
    p=tmp_path/"bad.jsonl"; p.write_text('{"query":"x"}\n',encoding="utf-8")
    with pytest.raises(ValueError, match="query_id"): load_eval_jsonl(p)

def test_eval_dataset_loader_rejects_invalid_json_with_line_number(tmp_path):
    p=tmp_path/"bad.jsonl"; p.write_text('{bad}\n',encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"): load_eval_jsonl(p)

def _c(cid, doc, st, content): return EvidenceChunk(cid,doc,st,"fixture",st,content,{})
class FakeAgent:
    def run(self,q):
        st=AgentState(q); st.retrieval_route="sql" if "Nice" in q else "hybrid"; st.expected_answer_type="direct_field_answer"; st.answer="Class 12" if "Nice" in q else "Temu policy prohibits trademark infringement."
        ev=[_c("trademark:sn:90000001:trademark_class:12","trademark:sn:90000001","trademark","Class 12 MERCEDES"), _c("policy:temu:ip:trademark","policy:temu:ip","policy","Temu policy prohibits trademark infringement")]
        st.reranked_evidence=ev; st.citations=[e.chunk_id for e in ev]; st.trace=["fake"]; return st

def test_evaluate_agent_outputs_per_example_metrics():
    examples=load_eval_jsonl(ROOT/"tests/fixtures/eval/eval_queries.jsonl")[:1]; res,summary=evaluate_agent(FakeAgent(), examples)
    assert res[0].metrics["Recall@5"]==1.0 and summary.num_examples==1

def test_evaluate_agent_summary_contains_required_metrics():
    _,s=evaluate_agent(FakeAgent(), load_eval_jsonl(ROOT/"tests/fixtures/eval/eval_queries.jsonl")[:1])
    for k in ["Recall@5","MAP@10","RoutingAccuracy","SourceTypeAccuracyStrict","LatencyMsMean"]: assert k in s.metrics

def test_evaluate_agent_groups_by_query_type_and_route():
    _,s=evaluate_agent(FakeAgent(), load_eval_jsonl(ROOT/"tests/fixtures/eval/eval_queries.jsonl")[:2])
    assert s.by_query_type and s.by_route

def test_report_writer_outputs_json_csv_markdown(tmp_path):
    res,s=evaluate_agent(FakeAgent(), load_eval_jsonl(ROOT/"tests/fixtures/eval/eval_queries.jsonl")[:1]); paths=write_eval_report_bundle(res,s,tmp_path)
    assert all(Path(p).exists() for p in paths.values()); assert "Overview" in (tmp_path/"eval_summary.md").read_text()

def test_eval_cli_demo_writes_report_bundle(tmp_path):
    cp=subprocess.run([sys.executable,"scripts/09_run_eval.py","--eval-file","tests/fixtures/eval/eval_queries.jsonl","--output-dir",str(tmp_path),"--demo"],cwd=ROOT,text=True,capture_output=True)
    assert cp.returncode==0, cp.stderr; assert (tmp_path/"eval_results.json").exists()

def test_eval_cli_requires_backend_without_demo(tmp_path):
    cp=subprocess.run([sys.executable,"scripts/09_run_eval.py","--eval-file","tests/fixtures/eval/eval_queries.jsonl","--output-dir",str(tmp_path)],cwd=ROOT,text=True,capture_output=True)
    assert cp.returncode!=0 and "requires" in cp.stderr

def test_default_ablation_configs_include_required_experiments():
    names={c.name for c in default_ablation_configs()}
    assert {"bm25_only","hybrid_rrf","hybrid_rerank","no_reranker","without_policy","rrf_k_10","rrf_k_60"} <= names

def test_ablation_runner_changes_config_per_experiment():
    seen=[]
    def fac(c): seen.append(c); return FakeAgent()
    run_ablations(load_eval_jsonl(ROOT/"tests/fixtures/eval/eval_queries.jsonl")[:1], fac, configs=default_ablation_configs()[:3])
    assert len({(c.retrieval_mode,c.reranker_provider,c.rrf_k) for c in seen})>1

def test_ablation_runner_outputs_results():
    out=run_ablations(load_eval_jsonl(ROOT/"tests/fixtures/eval/eval_queries.jsonl")[:1], lambda c: FakeAgent(), configs=default_ablation_configs()[:2])
    assert len(out)==2 and "Recall@5" in out[0].summary_metrics

def test_ablation_report_writer_outputs_json_csv_markdown(tmp_path):
    out=run_ablations(load_eval_jsonl(ROOT/"tests/fixtures/eval/eval_queries.jsonl")[:1], lambda c: FakeAgent(), configs=default_ablation_configs()[:2])
    paths=write_ablation_report_bundle(out,tmp_path); assert all(Path(p).exists() for p in paths.values())

def test_ablation_cli_demo_writes_report_bundle(tmp_path):
    cp=subprocess.run([sys.executable,"scripts/10_run_ablation.py","--eval-file","tests/fixtures/eval/eval_queries.jsonl","--output-dir",str(tmp_path),"--demo","--experiments","bm25_only,without_policy"],cwd=ROOT,text=True,capture_output=True)
    assert cp.returncode==0, cp.stderr; assert (tmp_path/"ablation_results.json").exists()

def test_ablation_runner_does_not_write_dummy_all_zero_metrics():
    out=run_ablations(load_eval_jsonl(ROOT/"tests/fixtures/eval/eval_queries.jsonl")[:1], lambda c: FakeAgent(), configs=default_ablation_configs()[:1])
    vals=[v for v in out[0].summary_metrics.values() if isinstance(v,(int,float))]
    assert any(v != 0 for v in vals)

def test_stage7_scripts_no_longer_raise_not_implemented(tmp_path):
    for script,out in [("09_run_eval.py",tmp_path/"e"),("10_run_ablation.py",tmp_path/"a")]:
        cp=subprocess.run([sys.executable,f"scripts/{script}","--eval-file","tests/fixtures/eval/eval_queries.jsonl","--output-dir",str(out),"--demo"],cwd=ROOT,text=True,capture_output=True)
        assert cp.returncode==0 and "NotImplementedError" not in cp.stderr

def test_no_duplicate_module_paths_created():
    assert not (ROOT/"src/crossborder_agentic_rag/eval").exists()
    assert not (ROOT/"src/crossborder_agentic_rag/evaluations").exists()
    assert not (ROOT/"src/crossborder_agentic_rag/benchmark").exists()
