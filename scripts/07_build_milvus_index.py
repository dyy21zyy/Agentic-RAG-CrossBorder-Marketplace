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

def _duplicate_chunk_id_examples(chunks):
    counts=Counter(c.chunk_id for c in chunks)
    return [{"chunk_id": cid, "count": count} for cid, count in counts.most_common() if count > 1][:20]

def parse_args(argv=None):
    p=argparse.ArgumentParser(description="Build a Milvus vector index for EvidenceChunk JSONL input.")
    p.add_argument("--input",required=True)
    p.add_argument("--dry-run",action="store_true",help="Embed and validate chunks without contacting or inserting into Milvus.")
    p.add_argument("--collection-name",default=os.getenv("MILVUS_COLLECTION_NAME","ip_rag_collection"))
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--report",default="data/processed/milvus_report.json")
    p.add_argument("--allow-empty",action="store_true")
    p.add_argument("--embedding-provider",default=os.getenv("EMBEDDING_PROVIDER","fake"),help="Embedding provider to use: fake, openai-compatible, or local. Fake is allowed only with --dry-run.")
    g=p.add_mutually_exclusive_group()
    g.add_argument("--verify-count", dest="verify_count", action="store_true", default=True, help="Verify persisted Milvus row count after indexing (default).")
    g.add_argument("--no-verify-count", dest="verify_count", action="store_false", help="Skip persisted row-count verification.")
    return p.parse_args(argv)

def run(args)->None:
    inp=Path(args.input)
    if not inp.is_file(): raise FileNotFoundError(f"Input chunks JSONL does not exist: {inp}")
    chunks=read_chunks_jsonl(inp)
    if not chunks and not args.allow_empty: raise ValueError("Input chunks JSONL is empty; pass --allow-empty to continue.")
    provider=build_embedding_provider(args.embedding_provider)
    if not args.dry_run and isinstance(provider, FakeEmbeddingProvider):
        raise RuntimeError("Real Milvus mode requires real embeddings; FakeEmbeddingProvider is only allowed with --dry-run. Set --embedding-provider openai-compatible or local after configuring its dependencies.")
    duplicate_examples=_duplicate_chunk_id_examples(chunks)
    unique_chunk_ids=len({c.chunk_id for c in chunks})
    report={"input":str(inp),"collection_name":args.collection_name,"dry_run":args.dry_run,"chunks_seen":len(chunks),"unique_chunk_ids":unique_chunk_ids,"duplicate_chunk_ids":len(duplicate_examples),"duplicate_chunk_id_examples":duplicate_examples,"chunks_indexed":0,"embedding_provider":provider.__class__.__name__,"embedding_dim":0,"source_type_counts":dict(Counter(c.source_type for c in chunks)),"source_subtype_counts":dict(Counter(c.source_subtype for c in chunks)),"milvus_inserted_attempted":0,"milvus_inserted":0,"actual_row_count":None,"verify_count":args.verify_count,"warnings":[],"failed_chunks":[]}
    if duplicate_examples:
        report["warnings"].append(f"Input has duplicate chunk_id values: duplicate_id_count={len(duplicate_examples)}, duplicate_extra={len(chunks)-unique_chunk_ids}. Milvus primary-key upsert/deduplication may persist only unique chunk IDs.")
    if args.dry_run:
        indexed=0
        for i in range(0,len(chunks),args.batch_size):
            batch=chunks[i:i+args.batch_size]
            vectors=provider.embed_documents(_texts(batch))
            if vectors and not report["embedding_dim"]: report["embedding_dim"]=len(vectors[0])
            indexed+=len(batch)
        report["chunks_indexed"]=indexed; write_report(report,args.report); print(f"Dry run: embedded {indexed} chunks (dim={report['embedding_dim']}); Milvus not contacted and no vectors inserted."); return
    store=None; inserted=0
    for i in range(0,len(chunks),args.batch_size):
        batch=chunks[i:i+args.batch_size]
        end=i+len(batch)
        print(f"Processing batch {i//args.batch_size+1} ({i}-{end-1}) of {len(chunks)} chunks.")
        try:
            vectors=provider.embed_documents(_texts(batch))
        except Exception as exc:
            report["failed_chunks"].append({"start":i,"end":end,"error":str(exc)})
            write_report(report,args.report)
            raise
        if vectors and not report["embedding_dim"]:
            report["embedding_dim"]=len(vectors[0])
            store=MilvusChunkStore(uri=os.getenv("MILVUS_URI","http://localhost:19530"),token=os.getenv("MILVUS_TOKEN"),collection_name=args.collection_name,embedding_dim=report["embedding_dim"],overwrite=args.overwrite)
            store.connect(); store.ensure_collection(); store.create_indexes()
        if store is not None:
            try:
                inserted+=store.insert_chunks(batch,vectors)
            except Exception as exc:
                report["failed_chunks"].append({"start":i,"end":end,"error":str(exc)})
                report["chunks_indexed"]=inserted; report["milvus_inserted_attempted"]=inserted; report["milvus_inserted"]=inserted
                write_report(report,args.report)
                raise
            print(f"Inserted {inserted}/{len(chunks)} chunks into Milvus collection {args.collection_name}.")
    if store is None:
        store=MilvusChunkStore(uri=os.getenv("MILVUS_URI","http://localhost:19530"),token=os.getenv("MILVUS_TOKEN"),collection_name=args.collection_name,embedding_dim=0,overwrite=args.overwrite)
        store.connect(); store.ensure_collection(); store.create_indexes()
    store.flush(); report["chunks_indexed"]=inserted; report["milvus_inserted_attempted"]=inserted; report["milvus_inserted"]=inserted
    if args.verify_count:
        actual=store.row_count(); report["actual_row_count"]=actual
        if actual != unique_chunk_ids:
            msg=f"Milvus row count mismatch: actual_row_count={actual}, unique_chunk_ids={unique_chunk_ids}, chunks_seen={len(chunks)}"
            report["warnings"].append(msg); write_report(report,args.report); print(msg, file=sys.stderr); raise RuntimeError(msg)
    store.close(); write_report(report,args.report); print(f"Inserted {inserted} chunks into Milvus collection {args.collection_name} (actual_row_count={report['actual_row_count']}).")

def main(argv=None)->int:
    args=parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0
if __name__=="__main__": raise SystemExit(main())
