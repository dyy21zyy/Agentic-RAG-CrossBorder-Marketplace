from __future__ import annotations
import json, os, subprocess, sys, types
from pathlib import Path
import pytest
from crossborder_agentic_rag.retrieval.reranker import build_reranker, LocalCrossEncoderReranker
from crossborder_agentic_rag.retrieval.utils import dedupe_chunks, evidence_to_dict, summarize_source_counts
from crossborder_agentic_rag.storage.milvus_store import parse_metadata_json, MilvusChunkStore

ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/'tests/fixtures/retrieval/sample_chunks.jsonl'

def run(args, env=None):
    e=os.environ.copy(); e.pop('RAG_MILVUS_URI',None); e.pop('MILVUS_URI',None)
    if env: e.update(env)
    return subprocess.run(args,cwd=ROOT,text=True,capture_output=True,env=e)

def test_dedupe_utils_preserve_order_and_counts():
    chunks=[{'chunk_id':'a','source_type':'patent','source_subtype':'claim','title':'T','content':'one'}, {'chunk_id':'b','source_type':'patent','source_subtype':'claim','title':'T2','content':'two'}, {'chunk_id':'a','source_type':'trademark','source_subtype':'x','content':'dup'}]
    out=dedupe_chunks(chunks)
    assert [c['chunk_id'] for c in out]==['a','b']
    assert summarize_source_counts(out)=={'source_type_counts':{'patent':2},'source_subtype_counts':{'claim':2}}
    assert summarize_source_counts([])=={'source_type_counts':{},'source_subtype_counts':{}}

def test_evidence_to_dict_handles_missing_fields():
    d=evidence_to_dict({'content':'abcdef','score':'1.5'}, rank=2, preview_chars=3)
    assert d['rank']==2 and d['content_preview']=='abc' and d['score']==1.5 and d['metadata']=={}

def test_metadata_json_safe_parse():
    assert parse_metadata_json(None)=={}
    assert parse_metadata_json('')=={}
    assert parse_metadata_json('{"a":1}')=={'a':1}
    assert parse_metadata_json('{bad')=={'_metadata_parse_error': True}

def test_reranker_optional_paths(monkeypatch):
    assert build_reranker('noop').rerank('q', [], 1)==[]
    assert build_reranker('lexical')
    monkeypatch.setitem(sys.modules,'sentence_transformers',None)
    with pytest.raises(ImportError, match='working torch'):
        LocalCrossEncoderReranker()
    with pytest.raises(NotImplementedError, match='not supported'):
        build_reranker('unknown')

def test_script_help_commands():
    for script in ['scripts/test_llm_api.py','scripts/run_dense_query.py','scripts/run_hybrid_query.py']:
        cp=run([sys.executable, script, '--help'])
        assert cp.returncode==0, cp.stderr

def test_llm_missing_key_fails_clearly():
    cp=run([sys.executable,'scripts/test_llm_api.py'], env={'LLM_MODEL':'m','OPENAI_API_KEY':''})
    assert cp.returncode!=0 and 'OPENAI_API_KEY is required' in cp.stderr

def test_dense_missing_uri_and_empty_query_fail_clearly():
    cp=run([sys.executable,'scripts/run_dense_query.py','--query','x','--embedding-provider','fake'])
    assert cp.returncode!=0 and 'RAG_MILVUS_URI is required' in cp.stderr
    cp=run([sys.executable,'scripts/run_dense_query.py','--query','   '])
    assert cp.returncode!=0 and '--query must be non-empty' in cp.stderr

def test_hybrid_bm25_only_json_without_milvus():
    cp=run([sys.executable,'scripts/run_hybrid_query.py','--chunks-path',str(FIX),'--query','trademark removal','--mode','bm25_only','--top-k','2','--output-json'])
    assert cp.returncode==0, cp.stderr
    data=json.loads(cp.stdout)
    assert data['mode']=='bm25_only' and isinstance(data['hits'], list) and 'source_type_counts' in data

def test_hybrid_dense_without_uri_fails_clearly():
    cp=run([sys.executable,'scripts/run_hybrid_query.py','--chunks-path',str(FIX),'--query','x','--mode','dense_only','--embedding-provider','fake'])
    assert cp.returncode!=0 and 'RAG_MILVUS_URI is required' in cp.stderr

def test_env_example_required_keys_and_no_secret():
    text=(ROOT/'.env.example').read_text()
    for key in ['OPENAI_API_KEY=','OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1','LLM_MODEL=deepseek-ai/DeepSeek-V4-Pro','RAG_MILVUS_URI=','LOCAL_EMBEDDING_MODEL=','RERANKER_PROVIDER=noop']:
        assert key in text
    assert 'sk-' not in text

def test_milvus_lite_load_called_before_search(monkeypatch):
    calls=[]
    class Client:
        def __init__(self, **kw): pass
        def load_collection(self, collection_name): calls.append(('load', collection_name))
        def search(self, **kw): calls.append(('search', kw['collection_name'])); return [[{'entity': {'chunk_id':'c','doc_id':'d','source_type':'patent','source_subtype':'claim','title':'t','content':'c','metadata_json':'{bad'}, 'distance':0.2}]]
    fake=types.SimpleNamespace(MilvusClient=Client)
    monkeypatch.setitem(sys.modules,'pymilvus',fake)
    out=MilvusChunkStore('/tmp/test.db',None,'coll',2).dense_search([0.1,0.2], top_k=1)
    assert calls[:2]==[('load','coll'),('search','coll')]
    assert out[0].metadata=={'_metadata_parse_error': True}
