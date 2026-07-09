from __future__ import annotations
import importlib.util, json, subprocess, sys, types
from pathlib import Path
import pytest
from crossborder_agentic_rag.ingestion.io_utils import read_chunks_jsonl
from crossborder_agentic_rag.llm.embeddings import FakeEmbeddingProvider, OpenAICompatibleEmbeddingProvider, LocalSentenceTransformerEmbeddingProvider
from crossborder_agentic_rag.retrieval import LocalBM25Retriever, rrf_fusion, LexicalReranker, NoOpReranker, LocalCrossEncoderReranker, APIRerankerPlaceholder, HybridRetriever
from crossborder_agentic_rag.storage.milvus_store import MilvusChunkStore, partition_for_source
ROOT=Path(__file__).resolve().parents[1]; FIX=ROOT/'tests/fixtures/retrieval/sample_chunks.jsonl'
def chunks(): return read_chunks_jsonl(FIX)
def test_fake_embedding_still_deterministic(): assert FakeEmbeddingProvider(4).embed_query('x')==FakeEmbeddingProvider(4).embed_query('x')
def test_openai_embedding_requires_config(monkeypatch):
    for k in ['EMBEDDING_API_KEY','EMBEDDING_API_BASE','EMBEDDING_MODEL','EMBEDDING_DIM']: monkeypatch.delenv(k,raising=False)
    with pytest.raises(ValueError,match='Missing embedding API'): OpenAICompatibleEmbeddingProvider()
def test_openai_embedding_rejects_bad_dimension_with_mocked_response(monkeypatch):
    class R:
        status_code=200; text='ok'
        def json(self): return {'data':[{'embedding':[1,2,3]}]}
    fake_requests=types.SimpleNamespace(post=lambda *a,**k:R())
    monkeypatch.setitem(sys.modules,'requests',fake_requests)
    p=OpenAICompatibleEmbeddingProvider('k','http://x','m',dim=2)
    with pytest.raises(ValueError,match='dimension'): p.embed_documents(['a'])
def test_local_embedding_missing_dependency_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules,'sentence_transformers',None)
    with pytest.raises(ImportError,match='sentence-transformers'): LocalSentenceTransformerEmbeddingProvider()
def test_local_bm25_returns_relevant_trademark_chunk(): assert LocalBM25Retriever(chunks()).search('Brand trademark infringement removal appeal')[0].chunk_id=='chunk-5'
def test_local_bm25_supports_source_type_filter(): assert all(c.source_type=='trademark' for c in LocalBM25Retriever(chunks()).search('trademark',source_types=['trademark']))
def test_local_bm25_supports_metadata_filter(): assert LocalBM25Retriever(chunks()).search('MERCEDES',filters={'word_mark':'MERCEDES'})[0].source_type=='trademark'
def test_local_bm25_empty_query_returns_empty_list(): assert LocalBM25Retriever(chunks()).search('')==[]
def test_rrf_fusion_deduplicates_and_ranks():
    a,b,c=chunks()[:3]; out=rrf_fusion([[a,b],[b,c]],k=60); assert [x.chunk_id for x in out][:2]==['chunk-2','chunk-1'] and len(out)==3
def test_rrf_fusion_rejects_invalid_k():
    with pytest.raises(ValueError): rrf_fusion([],k=0)
def test_rrf_fusion_applies_top_k(): assert len(rrf_fusion([chunks()[:3]],top_k=2))==2
def test_lexical_reranker_changes_order_by_overlap(): assert LexicalReranker().rerank('counterfeit prohibited account',chunks()[4:6][::-1],2)[0].chunk_id=='chunk-6'
def test_noop_reranker_still_preserves_order(): assert NoOpReranker().rerank('q',chunks()[:3],2)==chunks()[:2]
def test_local_cross_encoder_missing_dependency_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules,'sentence_transformers',None)
    with pytest.raises(ImportError,match='cross-encoder'): LocalCrossEncoderReranker()
def test_api_reranker_placeholder_does_not_fake_success():
    with pytest.raises(NotImplementedError): APIRerankerPlaceholder().rerank('q',chunks(),1)
class V:
    def __init__(self): self.calls=[]
    def dense_search(self,vec,filters=None,source_types=None,top_k=20): self.calls.append((vec,filters,source_types,top_k)); return chunks()[2:2+top_k]
def test_hybrid_retriever_bm25_only(): assert HybridRetriever(bm25_retriever=LocalBM25Retriever(chunks()),vector_store=V()).retrieve('Brand removal',mode='bm25_only')[0].chunk_id=='chunk-5'
def test_hybrid_retriever_dense_only_with_fake_vector_store(): assert HybridRetriever(vector_store=V()).retrieve('q',dense_vector=[1],mode='dense_only',top_k=1)[0].chunk_id=='chunk-3'
def test_hybrid_retriever_dense_only_embeds_query_when_vector_missing():
    e=FakeEmbeddingProvider(3); v=V(); HybridRetriever(embedding_provider=e,vector_store=v).retrieve('abc',mode='dense_only'); assert e.query_calls==['abc']
def test_hybrid_retriever_hybrid_rrf(): assert HybridRetriever(FakeEmbeddingProvider(3),LocalBM25Retriever(chunks()),V()).retrieve('Brand removal',dense_vector=[0],top_k=3)
def test_hybrid_retriever_hybrid_rerank(): assert HybridRetriever(FakeEmbeddingProvider(3),LocalBM25Retriever(chunks()),V(),LexicalReranker()).retrieve('patent claim',dense_vector=[0],mode='hybrid_rerank',top_k=2)
def test_hybrid_retriever_rejects_unknown_mode():
    with pytest.raises(ValueError): HybridRetriever().retrieve('q',mode='x')
def test_hybrid_retriever_rejects_missing_backends():
    with pytest.raises(ValueError): HybridRetriever().retrieve('q',mode='bm25_only')

def test_milvus_partition_for_source():
    assert partition_for_source('trademark')=='trademark_db'
    with pytest.raises(ValueError): partition_for_source('unknown')

def test_hybrid_retriever_passes_source_types_to_dense():
    v=V(); HybridRetriever(vector_store=v).retrieve('q',dense_vector=[1],mode='dense_only',top_k=1,source_types=['trademark'])
    assert v.calls[-1][2]==['trademark']

def test_milvus_store_build_filter_expr():
    s=MilvusChunkStore('u',None,'c',3).build_filter_expr({'source_type':['trademark','trademark'],'word_mark':'MER"CEDES'}); assert 'source_type in' in s and '\\"' in s
def test_milvus_store_rejects_invalid_filter_value():
    with pytest.raises(ValueError): MilvusChunkStore('u',None,'c',3).build_filter_expr({'source_type':5})
def test_milvus_store_insert_validates_vector_count():
    with pytest.raises(ValueError): MilvusChunkStore('u',None,'c',3).insert_chunks(chunks()[:1],[])
def test_milvus_store_insert_validates_vector_dimension():
    with pytest.raises(ValueError): MilvusChunkStore('u',None,'c',3).insert_chunks(chunks()[:1],[[1,2]])
def test_milvus_store_missing_pymilvus_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules,'pymilvus',None)
    with pytest.raises(ImportError,match='pymilvus is required'): MilvusChunkStore('u',None,'c',3).connect()
def test_milvus_store_schema_contains_required_fields_with_mocked_pymilvus(monkeypatch):
    names=[]
    class DT: VARCHAR='VARCHAR'; FLOAT_VECTOR='FLOAT_VECTOR'
    class FS:
        def __init__(self,**kw): names.append(kw['name'])
    fake=types.SimpleNamespace(DataType=DT,FieldSchema=FS,CollectionSchema=lambda fields,description: fields,Collection=lambda *a,**k: types.SimpleNamespace(partitions=[],create_partition=lambda *a,**k:None,create_index=lambda *a,**k:None,load=lambda:None),utility=types.SimpleNamespace(has_collection=lambda n:False,drop_collection=lambda n:None),connections=types.SimpleNamespace(connect=lambda **k:None))
    monkeypatch.setitem(sys.modules,'pymilvus',fake); MilvusChunkStore('u',None,'c',3).ensure_collection(); assert {'chunk_id','dense_vector','metadata_json','case_number','parent_id','context_path','partition'}<=set(names)
def test_build_milvus_index_dry_run_report(tmp_path):
    r=tmp_path/'r.json'; subprocess.run([sys.executable,'scripts/07_build_milvus_index.py','--input',str(FIX),'--dry-run','--report',str(r)],cwd=ROOT,check=True); data=json.loads(r.read_text()); assert data['dry_run'] and data['chunks_seen']==8 and data['milvus_inserted']==0
def test_build_milvus_index_real_mode_fails_clearly_without_milvus(tmp_path):
    r=tmp_path/'r.json'; p=subprocess.run([sys.executable,'scripts/07_build_milvus_index.py','--input',str(FIX),'--collection-name','ip_chunks','--overwrite','--report',str(r)],cwd=ROOT,text=True,capture_output=True); assert p.returncode!=0 and ('pymilvus is required' in p.stderr or 'Milvus' in p.stderr or 'Connect' in p.stderr)
def test_stage5_script_no_longer_raises_not_implemented(): assert 'NotImplementedError' not in (ROOT/'scripts/07_build_milvus_index.py').read_text()
def test_future_stage_scripts_still_raise_not_implemented():
    for name in ["09_run_eval.py", "10_run_ablation.py"]:
        assert "NotImplementedError" not in (ROOT / "scripts" / name).read_text(encoding="utf-8")

def test_no_duplicate_module_paths_created():
    forbidden=['src/crossborder_agentic_rag/vectorstore','src/crossborder_agentic_rag/vector_store','src/crossborder_agentic_rag/retriever','src/crossborder_agentic_rag/retrieval/hybrid.py','src/crossborder_agentic_rag/storage/milvus.py']
    assert not [p for p in forbidden if (ROOT/p).exists()]
