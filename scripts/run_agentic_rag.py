"""Run Agentic RAG or the explicit non-agent basic RAG baseline."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from agentic_rag_cli_common import load_env, print_result, run_pipeline


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-mode", choices=["agentic", "basic_rag"], default="agentic")
    parser.add_argument("--query", required=True)
    parser.add_argument("--duckdb-path")
    parser.add_argument("--chunks-path")
    parser.add_argument("--use-milvus", action="store_true")
    parser.add_argument("--collection-name", default=os.getenv("MILVUS_COLLECTION_NAME", "ip_chunks"))
    parser.add_argument("--embedding-provider", default=os.getenv("EMBEDDING_PROVIDER", "fake"))
    parser.add_argument("--retrieval-mode", choices=["bm25_only", "dense_only", "hybrid_rrf", "hybrid_rerank"], default="bm25_only")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--reranker-provider", choices=["noop", "none", "lexical", "local", "cross_encoder", "cross-encoder"], default=os.getenv("RERANKER_PROVIDER", "noop"))
    parser.add_argument("--reranker-model")
    parser.add_argument("--source-types", default="trademark,patent,litigation")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-provider", default=os.getenv("LLM_PROVIDER", "template"))
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument("--output-json", action="store_true")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    load_env()
    args = parse_args(argv)
    result = run_pipeline(args.query, args)
    print_result(result, args.output_json, args.show_trace)
    return result


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
