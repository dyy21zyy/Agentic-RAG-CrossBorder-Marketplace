#!/usr/bin/env python3
"""Build a lightweight NetworkX GraphRAG index from EvidenceChunk JSONL."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from crossborder_agentic_rag.graph.graph_builder import build_graph_from_chunks
from crossborder_agentic_rag.graph.graph_store import save_graph
from crossborder_agentic_rag.ingestion.io_utils import read_chunks_jsonl

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build lightweight NetworkX GraphRAG index from EvidenceChunk JSONL.")
    p.add_argument("--chunks-path", required=True)
    p.add_argument("--output-dir", default="data/processed/graph")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--allow-empty", action="store_true")
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    chunks = read_chunks_jsonl(args.chunks_path)
    if args.limit is not None: chunks = chunks[:args.limit]
    if not chunks and not args.allow_empty:
        print("No chunks found; pass --allow-empty to build an empty graph.", file=sys.stderr)
        return 2
    graph = build_graph_from_chunks(chunks)
    paths = save_graph(graph, args.output_dir)
    print(f"chunks_seen={len(chunks)}")
    print(f"graph_nodes={graph.number_of_nodes()}")
    print(f"graph_edges={graph.number_of_edges()}")
    for name, path in paths.items(): print(f"{name}={path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
