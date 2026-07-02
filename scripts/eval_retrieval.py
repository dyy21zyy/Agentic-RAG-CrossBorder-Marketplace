#!/usr/bin/env python
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path: sys.path.insert(0,str(ROOT/"src"))
from crossborder_agentic_rag.evaluation.dataset import load_eval_dataset
from crossborder_agentic_rag.evaluation import retrieval_metrics as rm
from crossborder_agentic_rag.evaluation.reporting import write_jsonl,write_json,write_csv,write_markdown_summary,aggregate_metric_rows
from crossborder_agentic_rag.ingestion.io_utils import read_chunks_jsonl
from crossborder_agentic_rag.llm.embeddings import FakeEmbeddingProvider, build_embedding_provider
from crossborder_agentic_rag.retrieval import HybridRetriever, LocalBM25Retriever, build_reranker

class DemoVectorStore:
    def __init__(self,chunks): self.chunks=chunks
    def dense_search(self,dense_vector,filters=None,source_types=None,top_k=20): return self.chunks[:top_k]

def hitdict(c):
    d=c.to_dict(); d["content_preview"]=d.pop("content")[:240]; return d

def metrics(hits, ex, ks):
    out={}
    for k in ks:
        out[f"Precision@{k}"]=rm.precision_at_k(hits,ex,k); out[f"Recall@{k}"]=rm.recall_at_k(hits,ex,k); out[f"HitRate@{k}"]=rm.hit_rate_at_k(hits,ex,k); out[f"MRR@{k}"]=rm.mrr_at_k(hits,ex,k); out[f"nDCG@{k}"]=rm.ndcg_at_k(hits,ex,k); out[f"SourceTypeCoverage@{k}"]=rm.source_type_coverage(hits,ex.expected_source_types,k); out[f"SourceSubtypeCoverage@{k}"]=rm.source_subtype_coverage(hits,ex.expected_source_subtypes,k)
    return out

def main(argv=None):
    p=argparse.ArgumentParser(description="Evaluate retrieval modes without LLM answering.")
    p.add_argument("--eval-path",default="eval/queries_small.jsonl"); p.add_argument("--chunks-path",default="data/processed/chunks_qa_300k.jsonl"); p.add_argument("--collection-name",default="ip_chunks_qa_300k")
    p.add_argument("--modes",default="bm25_only,hybrid_rrf,hybrid_rerank"); p.add_argument("--top-k-values",default="5,8,10"); p.add_argument("--candidate-k",type=int,default=50); p.add_argument("--reranker-provider",default="lexical"); p.add_argument("--reranker-model"); p.add_argument("--embedding-provider",default="local"); p.add_argument("--embedding-model"); p.add_argument("--output-dir",default="reports/eval_retrieval"); p.add_argument("--limit",type=int); p.add_argument("--output-jsonl-name",default="retrieval_results.jsonl")
    a=p.parse_args(argv); examples=load_eval_dataset(a.eval_path)[:a.limit] if a.limit else load_eval_dataset(a.eval_path)
    chunks=read_chunks_jsonl(a.chunks_path); bm25=LocalBM25Retriever(chunks); ks=[int(x) for x in a.top_k_values.split(',') if x]; maxk=max(ks)
    rows=[]; modes=[m.strip() for m in a.modes.split(',') if m.strip()]
    for mode in modes:
        if mode != "bm25_only" and not (os.getenv("RAG_MILVUS_URI") or os.getenv("MILVUS_URI")):
            # Offline fixture fallback keeps CI Milvus-free while preserving the warning in config.
            embedding=FakeEmbeddingProvider(); vector=DemoVectorStore(chunks)
        else:
            embedding=FakeEmbeddingProvider() if mode=="bm25_only" else build_embedding_provider(a.embedding_provider); vector=DemoVectorStore(chunks) if mode!="bm25_only" else None
        reranker=build_reranker(a.reranker_provider,a.reranker_model) if mode=="hybrid_rerank" else None
        retriever=HybridRetriever(embedding,bm25,vector,reranker)
        for ex in examples:
            st=time.perf_counter(); dense=embedding.embed_query(ex.query) if embedding and mode!="bm25_only" else None
            hits=retriever.retrieve(ex.query,dense_vector=dense,top_k=maxk,source_types=None,mode=mode,candidate_k=a.candidate_k); lat=(time.perf_counter()-st)*1000
            rows.append({"id":ex.id,"query":ex.query,"mode":mode,"top_k":maxk,"candidate_k":a.candidate_k,"latency_ms":lat,"hits":[hitdict(c) for c in hits],"metrics":metrics(hits,ex,ks)})
    out=Path(a.output_dir); write_jsonl(out/a.output_jsonl_name,rows)
    summary={"run_config":vars(a),"groups":aggregate_metric_rows(rows,"mode")}; write_json(out/"retrieval_summary.json",summary)
    flat=[]
    for g,ms in summary["groups"].items():
        for m,v in ms.items(): flat.append({"mode":g,"metric":m,**(v if isinstance(v,dict) else {"mean":v})})
    write_csv(out/"retrieval_summary.csv",flat); write_markdown_summary(out/"retrieval_summary.md",summary)
if __name__=="__main__": main()
