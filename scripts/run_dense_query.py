#!/usr/bin/env python
"""Run a dense vector query against a Milvus collection."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crossborder_agentic_rag.llm.embeddings import build_embedding_provider
from crossborder_agentic_rag.storage.milvus_store import MilvusChunkStore
from crossborder_agentic_rag.retrieval.utils import dedupe_chunks, evidence_to_dict, summarize_source_counts

def get_milvus_uri() -> str | None: return os.getenv("RAG_MILVUS_URI") or os.getenv("MILVUS_URI")
def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--query", required=True); p.add_argument("--collection-name", default="ip_chunks_qa_300k"); p.add_argument("--top-k", type=int, default=10); p.add_argument("--embedding-provider", default=os.getenv("EMBEDDING_PROVIDER","local")); p.add_argument("--embedding-model"); p.add_argument("--output-json", action="store_true"); p.add_argument("--preview-chars", type=int, default=500); return p.parse_args()
def build_output(args, uri, hits):
    counts=summarize_source_counts(hits)
    return {"query":args.query,"collection_name":args.collection_name,"top_k":args.top_k,"milvus_uri":uri,"embedding_provider":args.embedding_provider,"hits":[evidence_to_dict(c,i+1,args.preview_chars) for i,c in enumerate(hits)],**counts}
def main() -> int:
    args=parse_args(); q=args.query.strip()
    if not q: print("--query must be non-empty.", file=sys.stderr); return 2
    uri=get_milvus_uri()
    if not uri: print("RAG_MILVUS_URI is required for dense retrieval. Set RAG_MILVUS_URI=/path/to/milvus.db (or MILVUS_URI for backward compatibility).", file=sys.stderr); return 2
    if args.embedding_model: os.environ["LOCAL_EMBEDDING_MODEL"]=args.embedding_model
    try:
        provider=build_embedding_provider(args.embedding_provider); vec=provider.embed_query(q)
        store=MilvusChunkStore(uri, os.getenv("MILVUS_TOKEN"), args.collection_name, len(vec)); hits=dedupe_chunks(store.dense_search(vec, top_k=args.top_k))
    except Exception as exc:
        print(f"Dense retrieval failed: {exc}", file=sys.stderr); return 1
    out=build_output(args, uri, hits)
    if args.output_json: print(json.dumps(out, ensure_ascii=False, indent=2)); return 0
    print(f"Query: {q}\nCollection: {args.collection_name}\nHits: {len(hits)}")
    for h in out["hits"]: print(f"\n[{h['rank']}] {h['title']} score={h['score']}\n{h['content_preview']}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
