#!/usr/bin/env python
"""Build a Milvus vector index for EvidenceChunk JSONL input."""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from crossborder_agentic_rag.ingestion.io_utils import read_chunks_jsonl, write_report
from crossborder_agentic_rag.llm.embeddings import FakeEmbeddingProvider, build_embedding_provider
from crossborder_agentic_rag.storage.milvus_store import MilvusChunkStore

def _texts(chunks): return [c.title+"\n"+c.content for c in chunks]

def parse_args(argv=None):
    p=argparse.ArgumentParser(description="Build a Milvus vector index for EvidenceChunk JSONL input.")
    p.add_argument("--input",required=True)
    p.add_argument("--dry-run",action="store_true",help="Embed and validate chunks without contacting or inserting into Milvus.")
    p.add_argument("--collection-name",default=os.getenv("MILVUS_COLLECTION_NAME","ip_chunks"))
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--report",default="data/processed/milvus_report.json")
    p.add_argument("--allow-empty",action="store_true")
    p.add_argument("--embedding-provider",default=os.getenv("EMBEDDING_PROVIDER","fake"),help="Embedding provider to use: fake, openai-compatible, or local. Fake is allowed only with --dry-run.")
    return p.parse_args(argv)

def run(args)->None:
    inp=Path(args.input)
    if not inp.is_file(): raise FileNotFoundError(f"Input chunks JSONL does not exist: {inp}")
    chunks=read_chunks_jsonl(inp)
    if not chunks and not args.allow_empty: raise ValueError("Input chunks JSONL is empty; pass --allow-empty to continue.")
    provider=build_embedding_provider(args.embedding_provider)
    if not args.dry_run and isinstance(provider, FakeEmbeddingProvider):
        raise RuntimeError("Real Milvus mode requires real embeddings; FakeEmbeddingProvider is only allowed with --dry-run. Set --embedding-provider openai-compatible or local after configuring its dependencies.")
    vectors=[]
    for i in range(0,len(chunks),args.batch_size): vectors.extend(provider.embed_documents(_texts(chunks[i:i+args.batch_size])))
    dim=len(vectors[0]) if vectors else 0
    report={"input":str(inp),"collection_name":args.collection_name,"dry_run":args.dry_run,"chunks_seen":len(chunks),"chunks_indexed":0,"embedding_provider":provider.__class__.__name__,"embedding_dim":dim,"source_type_counts":dict(Counter(c.source_type for c in chunks)),"source_subtype_counts":dict(Counter(c.source_subtype for c in chunks)),"milvus_inserted":0,"warnings":[],"failed_chunks":[]}
    if args.dry_run:
        report["chunks_indexed"]=len(chunks); write_report(report,args.report); print(f"Dry run: embedded {len(chunks)} chunks (dim={dim}); Milvus not contacted and no vectors inserted."); return
    store=MilvusChunkStore(uri=os.getenv("MILVUS_URI","http://localhost:19530"),token=os.getenv("MILVUS_TOKEN"),collection_name=args.collection_name,embedding_dim=dim,overwrite=args.overwrite)
    store.connect(); store.ensure_collection(); store.create_indexes(); inserted=0
    for i in range(0,len(chunks),args.batch_size): inserted+=store.insert_chunks(chunks[i:i+args.batch_size],vectors[i:i+args.batch_size])
    store.flush(); report["chunks_indexed"]=inserted; report["milvus_inserted"]=inserted; write_report(report,args.report); print(f"Inserted {inserted} chunks into Milvus collection {args.collection_name}.")

def main(argv=None)->int:
    args=parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0
if __name__=="__main__": raise SystemExit(main())
