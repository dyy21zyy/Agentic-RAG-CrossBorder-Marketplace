"""Interactive chat for Agentic RAG or the explicit non-agent baseline."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
from agentic_rag_cli_common import load_env, print_result, build_runtime

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline-mode", choices=["agentic","basic_rag"], default="agentic"); p.add_argument("--duckdb-path"); p.add_argument("--chunks-path"); p.add_argument("--use-milvus", action="store_true"); p.add_argument("--collection-name", default=os.getenv("MILVUS_COLLECTION_NAME","ip_chunks_qa_300k")); p.add_argument("--embedding-provider", default=os.getenv("EMBEDDING_PROVIDER","fake")); p.add_argument("--retrieval-mode", choices=["bm25_only","dense_only","hybrid_rrf","hybrid_rerank"], default="bm25_only"); p.add_argument("--candidate-k", type=int, default=50); p.add_argument("--top-k", type=int, default=8); p.add_argument("--max-iterations", type=int, default=2); p.add_argument("--reranker-provider", choices=["noop","none","lexical","local","cross_encoder","cross-encoder"], default=os.getenv("RERANKER_PROVIDER","lexical")); p.add_argument("--reranker-model"); p.add_argument("--source-types", default="trademark,patent,litigation"); p.add_argument("--use-llm", action="store_true"); p.add_argument("--llm-provider", default=os.getenv("LLM_PROVIDER","template")); p.add_argument("--llm-model", default=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL")); p.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")); p.add_argument("--max-evidence-for-llm", type=int, default=6); p.add_argument("--max-chars-per-evidence", type=int, default=450); p.add_argument("--llm-max-tokens", type=int, default=800); p.add_argument("--temperature", type=float, default=0.0); p.add_argument("--show-sources", action="store_true"); p.add_argument("--show-trace", action="store_true"); p.add_argument("--output-json", action="store_true"); p.add_argument("--demo", action="store_true"); return p.parse_args(argv)

def main(argv=None):
    load_env(); args=parse_args(argv); runtime=build_runtime(args)
    print("Agentic RAG interactive chat"); print(f"pipeline_mode: {args.pipeline_mode}"); print(f"retrieval_mode: {args.retrieval_mode}"); print(f"reranker: {args.reranker_provider}")
    if args.use_llm: print(f"LLM provider: {args.llm_provider}")
    try:
        while True:
            query=input("User > ").strip()
            if query.lower() in {"exit","quit","q"}: break
            if not query: continue
            try: print_result(runtime.run_query(query), args.output_json, args.show_trace, args.show_sources)
            except Exception as exc: print(f"Error: {exc}", file=sys.stderr)
    except KeyboardInterrupt: print("\nExiting.")
if __name__=="__main__":
    try: main()
    except Exception as exc: print(f"Error: {exc}", file=sys.stderr); sys.exit(1)
