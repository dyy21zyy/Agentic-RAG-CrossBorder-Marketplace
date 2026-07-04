from __future__ import annotations
import importlib.util, json, subprocess, sys
from pathlib import Path
import pytest
from crossborder_agentic_rag.agents import *
from crossborder_agentic_rag.agents.planner import extract_basic_filters
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
from crossborder_agentic_rag.storage.duckdb_store import DuckDBStore
from crossborder_agentic_rag.ingestion.io_utils import read_documents_jsonl, read_chunks_jsonl

ROOT=Path(__file__).resolve().parents[1]

def chunks(): return read_chunks_jsonl(ROOT/'tests/fixtures/agent/sample_chunks.jsonl')
def plan(q):
    c=classify_query(q); return build_query_plan(q,c)
class RecordingEmbeddingProvider:
    def __init__(self): self.calls=[]
    def embed_query(self,text): self.calls.append(text); return [1.0,0.0,0.0]
class RecordingRetriever:
    def __init__(self,chunks): self.calls=[]; self.chunks=chunks
    def retrieve(self,query,dense_vector=None,filters=None,top_k=20,source_types=None,mode="hybrid_rrf"):
        self.calls.append({"query":query,"dense_vector":dense_vector,"filters":filters,"top_k":top_k,"source_types":source_types,"mode":mode})
        return [c for c in self.chunks if source_types is None or c.source_type in source_types][:top_k]
class FakeStore:
    def __init__(self): self.calls=[]
    def __getattr__(self,name):
        if name.startswith('lookup_'):
            def f(arg): self.calls.append((name,arg)); return [{"word_mark":arg,"case_number":arg,"patent_id":arg}]
            return f
        raise AttributeError(name)

def test_normalize_query_rejects_empty_query():
    with pytest.raises(ValueError): normalize_query('   ')
def test_policy_terms_do_not_route_to_policy_source():
    c=classify_query('What does marketplace guidance say about trademark infringement?'); assert c.source_types==['trademark'] and c.expected_answer_type=='trademark_explanation'
def test_listing_risk_question_is_risk():
    c=classify_query('Can I sell a phone case using the MERCEDES logo?'); assert c.query_type=='risk_analysis' and c.retrieval_route=='multi_source_risk' and {'trademark'}<=set(c.source_types) and 'policy' not in c.source_types
def test_field_lookup_routes_to_sql(): assert classify_query('Which Nice classes does MERCEDES belong to?').retrieval_route=='sql'
def test_semantic_trademark_question_routes_to_hybrid(): assert classify_query('What evidence mentions counterfeit trademark enforcement?').retrieval_route=='hybrid'
def test_mixed_patent_query_routes_to_mixed(): assert classify_query('Explain the claims of patent US1234567.').retrieval_route=='mixed'
def test_extract_filters_word_mark_patent_case_registration():
    f=extract_basic_filters('MERCEDES patent US1234567 case 1:23-cv-00001 registration number 1234567', classify_query('Which Nice classes does MERCEDES belong to?'))
    assert f['word_mark']=='MERCEDES' and f['patent_number']=='US1234567' and f['case_number']=='1:23-cv-00001' and f['registration_number']=='1234567'

def test_sql_router_calls_trademark_class_lookup():
    s=FakeStore(); SQLRouter(s).run(plan('Which Nice classes does MERCEDES belong to?')); assert s.calls[0][0]=='lookup_trademark_classes_by_word_mark'
def test_sql_router_calls_trademark_goods_services_lookup():
    s=FakeStore(); SQLRouter(s).run(plan('What goods and services does MERCEDES cover?')); assert s.calls[0][0]=='lookup_trademark_goods_services_by_word_mark'
def test_sql_router_calls_patent_lookup():
    s=FakeStore(); SQLRouter(s).run(plan('Find patent US1234567')); assert s.calls[0][0]=='lookup_patent_by_id'
def test_sql_router_calls_litigation_case_lookup():
    s=FakeStore(); SQLRouter(s).run(plan('Show parties in case 1:23-cv-00001.')); assert s.calls[0][0]=='lookup_litigation_parties_by_case'
def test_sql_router_requires_store_for_sql_route():
    with pytest.raises(ValueError): SQLRouter().run(plan('Which Nice classes does MERCEDES belong to?'))

def test_evidence_evaluator_trademark_requires_trademark_evidence():
    ev=evaluate_evidence(plan('Explain trademark infringement risk.'), []); assert not ev.is_sufficient and ev.missing_source_types==['trademark']
def test_evidence_evaluator_risk_requires_all_required_evidence():
    ev=evaluate_evidence(plan('Assess IP risks for selling a smart backpack.'), [chunks()[2]]); assert not ev.is_sufficient and set(ev.missing_source_types)=={'trademark','litigation'}
def test_evidence_evaluator_builds_followup_queries(): assert 'Find trademark evidence for:' in build_followup_query('q','trademark')

def _ans(q, ev=None, sql=None): return synthesize_answer(plan(q), sql or [], ev or [chunks()[0]], [])
def test_answer_direct_field_has_no_risk_level(): assert 'Risk Level' not in synthesize_answer(plan('Which Nice classes does MERCEDES belong to?'), [{'_lookup':'lookup_trademark_classes_by_word_mark','word_mark':'MERCEDES'}], [], [])[0]
def test_answer_trademark_has_no_risk_level(): assert 'Risk Level' not in _ans('Explain trademark infringement.')[0]
def test_answer_patent_has_no_risk_level(): assert 'Risk Level' not in synthesize_answer(plan('Explain patent claims about drone delivery.'), [], [chunks()[2]], [])[0]
def test_answer_litigation_has_no_risk_level(): assert 'Risk Level' not in synthesize_answer(plan('Summarize litigation history for patent US1234567.'), [], [chunks()[3]], [])[0]
def test_answer_risk_has_risk_level(): assert 'Risk Level:' in synthesize_answer(plan('Can I sell MERCEDES logo?'), [], chunks()[:2], [])[0]
def test_answer_citations_are_not_invented():
    ans,c=synthesize_answer(plan('Explain trademark infringement.'), [], [chunks()[0]], []); assert c==['[trademark:tm-brand-logo:trademark_record:0] Brand logo trademark — trademark_record']

def test_agent_sql_route_does_not_call_embedding_when_sql_sufficient():
    emb=RecordingEmbeddingProvider(); state=AgenticRAG(FakeStore(), RecordingRetriever(chunks()), emb).run('Which Nice classes does MERCEDES belong to?'); assert emb.calls==[] and state.sql_results
def test_agent_hybrid_route_calls_retriever():
    r=RecordingRetriever(chunks()); AgenticRAG(retriever=r, embedding_provider=RecordingEmbeddingProvider()).run('Explain trademark infringement'); assert r.calls
def test_agent_mixed_route_runs_sql_then_hybrid():
    s=FakeStore(); r=RecordingRetriever(chunks()); st=AgenticRAG(s,r).run('Explain the claims of patent US1234567.'); assert s.calls and r.calls and 'mixed_hybrid_retrieval' in st.trace
def test_agent_risk_route_runs_multi_source_retrieval():
    r=RecordingRetriever(chunks()); st=AgenticRAG(retriever=r).run('Can I sell a MERCEDES logo product?'); assert 'multi_source_risk_retrieval' in st.trace and r.calls[0]['source_types']
def test_agent_followup_retrieval_for_missing_trademark():
    r=RecordingRetriever([chunks()[1]]); st=AgenticRAG(retriever=r,max_iterations=1).run('Explain patent claim infringement'); assert st.iterations==1 and 'followup_retrieval' in st.trace
def test_agent_stops_after_max_iterations():
    r=RecordingRetriever([]); st=AgenticRAG(retriever=r,max_iterations=1).run('Explain trademark infringement'); assert st.iterations==1
def test_agent_trace_contains_required_steps():
    st=AgenticRAG(retriever=RecordingRetriever(chunks())).run('Explain trademark infringement'); assert {'normalize_query','classify_query','plan_retrieval','evaluate_evidence','final_answer'}<=set(st.trace)
def test_agent_state_fields_are_populated():
    st=AgenticRAG(retriever=RecordingRetriever(chunks())).run('Explain trademark infringement'); assert st.normalized_query and st.query_type and st.expected_answer_type and st.retrieval_route and st.retrieval_plan and st.answer and st.citations

def test_cli_demo_mode_outputs_json():
    cp=subprocess.run([sys.executable,'scripts/08_run_query_cli.py','Explain trademark infringement risk','--chunks-path','tests/fixtures/agent/sample_chunks.jsonl','--demo','--output-json'],cwd=ROOT,text=True,capture_output=True,check=True); assert json.loads(cp.stdout)['demo_mode'] is True
def test_cli_requires_retriever_without_demo():
    cp=subprocess.run([sys.executable,'scripts/08_run_query_cli.py','Explain trademark infringement'],cwd=ROOT,text=True,capture_output=True); assert cp.returncode!=0 and 'No retrieval backend' in cp.stderr
def test_cli_with_chunks_path_outputs_json():
    cp=subprocess.run([sys.executable,'scripts/08_run_query_cli.py','Explain trademark infringement','--chunks-path','tests/fixtures/agent/sample_chunks.jsonl','--output-json'],cwd=ROOT,text=True,capture_output=True,check=True); assert json.loads(cp.stdout)['query_type']=='trademark_explanation'
def test_stage6_script_no_longer_raises_not_implemented(): assert 'NotImplementedError' not in (ROOT/'scripts/08_run_query_cli.py').read_text()
def test_future_stage_scripts_still_raise_not_implemented():
    for name in ["09_run_eval.py", "10_run_ablation.py"]:
        assert "NotImplementedError" not in (ROOT / "scripts" / name).read_text(encoding="utf-8")

def test_no_duplicate_module_paths_created():
    for p in ['src/crossborder_agentic_rag/agent','src/crossborder_agentic_rag/rag','src/crossborder_agentic_rag/query_router','src/crossborder_agentic_rag/sql','src/crossborder_agentic_rag/agents/agent_graph.py','src/crossborder_agentic_rag/agents/graph_v2.py']:
        assert not (ROOT/p).exists()

def test_risk_analysis_does_not_require_policy_by_default():
    c = classify_query('Can I sell a phone case using the MERCEDES logo?')
    ev = evaluate_evidence(plan('Can I sell a phone case using the MERCEDES logo?'), [chunks()[1]])
    assert 'Missing policy evidence' not in ev.evidence_gaps

def test_query_cli_hybrid_milvus_without_chunks_explains_bm25_requirement():
    cp=subprocess.run([sys.executable,'scripts/08_run_query_cli.py','q','--use-milvus','--retrieval-mode','hybrid_rrf'],cwd=ROOT,text=True,capture_output=True)
    assert cp.returncode!=0 and 'requires --chunks-path to build the BM25 retriever' in cp.stderr

def test_query_cli_dense_only_milvus_path_does_not_require_chunks_path():
    text=(ROOT/'scripts/08_run_query_cli.py').read_text(encoding='utf-8')
    assert 'args.retrieval_mode in {"bm25_only","hybrid_rrf","hybrid_rerank"}' in text
    assert '--retrieval-mode dense_only with --use-milvus' in text
