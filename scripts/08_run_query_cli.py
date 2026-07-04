"""Run the Stage 6 Agentic RAG query CLI."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from crossborder_agentic_rag.agents.graph import AgenticRAG
from crossborder_agentic_rag.ingestion.io_utils import read_chunks_jsonl
from crossborder_agentic_rag.llm.embeddings import FakeEmbeddingProvider, build_embedding_provider
from crossborder_agentic_rag.retrieval.bm25 import LocalBM25Retriever
from crossborder_agentic_rag.retrieval.hybrid_retriever import HybridRetriever
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
from crossborder_agentic_rag.storage.duckdb_store import DuckDBStore
from crossborder_agentic_rag.storage.milvus_store import MilvusChunkStore

class BM25RetrieverAdapter:
    def __init__(self, chunks): self.bm25=LocalBM25Retriever(chunks)
    def retrieve(self, query, dense_vector=None, filters=None, top_k=20, source_types=None, mode="hybrid_rrf"):
        return self.bm25.search(query, filters=filters, source_types=source_types, top_k=top_k)

def _load_env():
    p=ROOT/".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def _demo_chunks():
    return [
        EvidenceChunk("trademark:brand-logo:0","brand-logo","trademark","trademark_record","Brand logo trademark","Trademark evidence describes infringement risk from unauthorized brand logos.",{"chunk_index":0}),
        EvidenceChunk("tm:mercedes:0","tm-mercedes","trademark","classes","MERCEDES trademark","MERCEDES registration number 1234567 covers Nice class 12 vehicles and related goods/services.",{"word_mark":"MERCEDES","registration_number":"1234567","chunk_index":0}),
        EvidenceChunk("pat:us1234567:0","pat-us1234567","patent","claims","US1234567 Patent","US1234567 includes a claim for drone delivery control and related drawing description.",{"patent_number":"US1234567","chunk_index":0}),
        EvidenceChunk("lit:case:0","lit-case","litigation","docket","Patent litigation case","Case 1:23-cv-00001 lists parties Example Corp v Seller and docket summary for patent US1234567.",{"case_number":"1:23-cv-00001","patent_number":"US1234567","chunk_index":0}),
    ]

def parse_args(argv=None):
    p=argparse.ArgumentParser(description="Run Stage 6 Agentic RAG query workflow")
    p.add_argument("query", nargs="?"); p.add_argument("--duckdb-path"); p.add_argument("--chunks-path"); p.add_argument("--use-milvus", action="store_true"); p.add_argument("--collection-name", default=os.getenv("MILVUS_COLLECTION_NAME","ip_chunks")); p.add_argument("--embedding-provider", default=os.getenv("EMBEDDING_PROVIDER","fake")); p.add_argument("--retrieval-mode", default="bm25_only"); p.add_argument("--top-k", type=int, default=20); p.add_argument("--max-iterations", type=int, default=2); p.add_argument("--demo", action="store_true"); p.add_argument("--output-json", action="store_true")
    return p.parse_args(argv)

def _compact(c):
    d=c.to_dict(); return {"chunk_id":d["chunk_id"],"doc_id":d["doc_id"],"source_type":d["source_type"],"source_subtype":d["source_subtype"],"title":d["title"],"score":d["score"],"metadata":d["metadata"],"content_preview":d["content"][:240]}

def main(argv=None):
    _load_env(); args=parse_args(argv)
    if not args.query: raise ValueError("query is required")
    chunks=[]; retriever=None; embedding=None; warnings=[]
    if args.demo:
        chunks=_demo_chunks(); embedding=FakeEmbeddingProvider(); retriever=BM25RetrieverAdapter(chunks)
    elif args.chunks_path:
        path=Path(args.chunks_path)
        if not path.is_file(): raise FileNotFoundError(f"Chunks path does not exist: {path}")
        chunks=read_chunks_jsonl(path); embedding=build_embedding_provider(args.embedding_provider)
        if args.use_milvus:
            dim=len(embedding.embed_query("dimension probe")); store=MilvusChunkStore(os.getenv("MILVUS_URI","http://localhost:19530"), os.getenv("MILVUS_TOKEN"), args.collection_name, dim); store.connect(); store.ensure_collection(); retriever=HybridRetriever(embedding_provider=embedding,bm25_retriever=LocalBM25Retriever(chunks), vector_store=store)
        else:
            retriever=BM25RetrieverAdapter(chunks) if args.retrieval_mode=="bm25_only" else HybridRetriever(embedding_provider=embedding,bm25_retriever=LocalBM25Retriever(chunks))
    elif args.use_milvus:
        if args.retrieval_mode in {"bm25_only","hybrid_rrf","hybrid_rerank"}:
            raise RuntimeError(f"retrieval-mode {args.retrieval_mode!r} requires --chunks-path to build the BM25 retriever; use --retrieval-mode dense_only with --use-milvus to query Milvus without local chunks.")
        embedding=build_embedding_provider(args.embedding_provider); dim=len(embedding.embed_query("dimension probe")); store=MilvusChunkStore(os.getenv("MILVUS_URI","http://localhost:19530"), os.getenv("MILVUS_TOKEN"), args.collection_name, dim); store.connect(); store.ensure_collection(); retriever=HybridRetriever(embedding_provider=embedding, vector_store=store)
    else:
        raise RuntimeError("No retrieval backend configured. Provide --chunks-path or --use-milvus, or pass --demo explicitly.")
    duck=None
    if args.duckdb_path:
        if not Path(args.duckdb_path).exists(): raise FileNotFoundError(f"DuckDB path does not exist: {args.duckdb_path}")
        duck=DuckDBStore(args.duckdb_path)
    state=AgenticRAG(duckdb_store=duck,retriever=retriever,embedding_provider=embedding if args.retrieval_mode!="bm25_only" and not args.demo else None,max_iterations=args.max_iterations,default_top_k=args.top_k).run(args.query)
    if args.output_json:
        print(json.dumps({"demo_mode": bool(args.demo), "query":state.query,"normalized_query":state.normalized_query,"query_type":state.query_type,"expected_answer_type":state.expected_answer_type,"retrieval_route":state.retrieval_route,"answer":state.answer,"citations":state.citations,"sql_results":state.sql_results,"source_chunks":[_compact(c) for c in state.retrieved_evidence],"evidence_gaps":state.evidence_gaps + warnings,"warnings":warnings,"iterations":state.iterations,"trace":state.trace}, ensure_ascii=False, indent=2))
    else:
        if args.demo: print("[demo mode]")
        print(state.answer)

if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr); sys.exit(1)
