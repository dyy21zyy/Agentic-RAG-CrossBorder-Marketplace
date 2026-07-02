from __future__ import annotations
import subprocess, sys
from pathlib import Path
import pytest
from crossborder_agentic_rag.retrieval import HybridRetriever, LexicalReranker, NoOpReranker, build_reranker
from crossborder_agentic_rag.retrieval.reranker import LocalCrossEncoderReranker
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
from crossborder_agentic_rag.schemas.results import AgentState

ROOT=Path(__file__).resolve().parents[1]

def chunk(i:int, title:str="", content:str="", score:float=1.0):
    return EvidenceChunk(f"c{i}",f"d{i}","patent","claim",title or f"Title {i}",content or "content",{},score)

class BM25:
    def __init__(self, hits): self.hits=hits; self.calls=[]
    def search(self,q,filters=None,source_types=None,top_k=20): self.calls.append(top_k); return self.hits[:top_k]
class Dense:
    def __init__(self, hits): self.hits=hits; self.calls=[]
    def dense_search(self,vec,filters=None,source_types=None,top_k=20): self.calls.append(top_k); return self.hits[:top_k]

def test_lexical_reranker_returns_top_k_and_improves_term_match():
    cands=[chunk(1,"Marketplace server","unrelated systems"),chunk(2,"Design patent bag","luggage ornamental design"),chunk(3,"Utility patent travel bag claim","smart travel bag luggage patent claim")]
    out=LexicalReranker().rerank("smart travel bag luggage patent claim", cands, 2)
    assert len(out)==2 and out[0].chunk_id=="c3" and out[1].chunk_id=="c2"
    assert out[0].metadata["reranker_provider"]=="lexical"

def test_lexical_reranker_does_not_mutate_original_candidate():
    c=chunk(1,"Travel bag","patent claim", score=0.5)
    out=LexicalReranker().rerank("travel patent", [c], 1)
    assert out[0] is not c
    assert c.metadata=={}
    assert out[0].metadata["reranker_score"] > 0

def test_local_reranker_empty_candidates_returns_without_model_call():
    r=LocalCrossEncoderReranker.__new__(LocalCrossEncoderReranker)
    r.model_name="fake"
    class Model:
        def predict(self, pairs): raise AssertionError("predict should not be called")
    r.model=Model()
    assert r.rerank("q", [], 10)==[]

def test_reranker_aliases_and_unknown_provider():
    assert isinstance(build_reranker("noop"), NoOpReranker)
    assert isinstance(build_reranker("none"), NoOpReranker)
    assert isinstance(build_reranker("lexical"), LexicalReranker)
    with pytest.raises((ValueError, NotImplementedError), match="not supported"):
        build_reranker("mystery")

def test_hybrid_rerank_uses_candidate_k_before_final_top_k():
    bm25=BM25([chunk(i,content=f"bm25 {i}") for i in range(1, 7)])
    dense=Dense([chunk(i,content=f"dense {i}") for i in range(4, 10)])
    retriever=HybridRetriever(bm25_retriever=bm25, vector_store=dense, reranker=NoOpReranker())
    out=retriever.retrieve("q", dense_vector=[0.1], top_k=2, mode="hybrid_rerank", candidate_k=5)
    assert len(out)==2
    assert bm25.calls==[5] and dense.calls==[5]
    assert len({c.chunk_id for c in out})==len(out)

def test_hybrid_rerank_requires_reranker():
    with pytest.raises(ValueError, match="Reranker is required"):
        HybridRetriever(bm25_retriever=BM25([]), vector_store=Dense([])).retrieve("q", dense_vector=[0.1], mode="hybrid_rerank")

def test_agent_state_phase2_defaults_are_independent_lists():
    a=AgentState("a"); b=AgentState("b")
    a.candidate_evidence.append(chunk(1))
    assert b.candidate_evidence==[] and a.retrieval_mode is None and a.candidate_k is None

def test_run_hybrid_query_help_works():
    cp=subprocess.run([sys.executable,"scripts/run_hybrid_query.py","--help"],cwd=ROOT,text=True,capture_output=True)
    assert cp.returncode==0 and "--candidate-k" in cp.stdout and "--reranker-provider" in cp.stdout

def test_run_hybrid_query_bm25_fixture_without_milvus():
    cp=subprocess.run([sys.executable,"scripts/run_hybrid_query.py","--query","smart travel bag","--chunks-path","tests/fixtures/agent/sample_chunks.jsonl","--mode","bm25_only","--top-k","2","--output-json"],cwd=ROOT,text=True,capture_output=True)
    assert cp.returncode==0, cp.stderr
    assert '"mode": "bm25_only"' in cp.stdout

def test_run_hybrid_query_hybrid_missing_milvus_fails_clearly(monkeypatch):
    monkeypatch.delenv("RAG_MILVUS_URI", raising=False); monkeypatch.delenv("MILVUS_URI", raising=False)
    cp=subprocess.run([sys.executable,"scripts/run_hybrid_query.py","--query","q","--chunks-path","tests/fixtures/agent/sample_chunks.jsonl","--mode","hybrid_rrf"],cwd=ROOT,text=True,capture_output=True)
    assert cp.returncode!=0 and "RAG_MILVUS_URI is required" in cp.stderr
