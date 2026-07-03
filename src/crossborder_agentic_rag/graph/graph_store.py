"""Persistence helpers for lightweight GraphRAG indexes."""
from __future__ import annotations
import json, pickle
from pathlib import Path
import networkx as nx

def save_graph(graph: nx.MultiDiGraph, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    nodes_path = out / "graph_nodes.jsonl"; edges_path = out / "graph_edges.jsonl"; graph_path = out / "ip_graph.pkl"
    with nodes_path.open("w", encoding="utf-8") as f:
        for node_id, data in graph.nodes(data=True):
            f.write(json.dumps({"node_id": node_id, **data}, ensure_ascii=False, default=str) + "\n")
    with edges_path.open("w", encoding="utf-8") as f:
        for source, target, _key, data in graph.edges(keys=True, data=True):
            f.write(json.dumps({"source": source, "target": target, **data}, ensure_ascii=False, default=str) + "\n")
    with graph_path.open("wb") as f:
        pickle.dump(graph, f)
    return {"nodes": str(nodes_path), "edges": str(edges_path), "graph": str(graph_path)}

def load_graph(path: str | Path) -> nx.MultiDiGraph:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GraphRAG index not found: {p}")
    with p.open("rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError(f"Expected networkx.MultiDiGraph in {p}")
    return graph
