"""Lightweight GraphRAG retriever over a NetworkX graph."""
from __future__ import annotations
import re
from pathlib import Path
import networkx as nx
from crossborder_agentic_rag.graph.graph_store import load_graph

STOPWORDS = {"the","and","for","with","from","that","this","are","was","were","what","when","where","which","about","into","over","under","between","against"}
KEEP = {"bag","luggage","backpack","suitcase","patent","trademark","litigation","gps","lock"}
EMPTY = {"matched_entities": [], "related_nodes": [], "related_edges": [], "related_doc_ids": [], "related_chunk_ids": []}

class GraphRetriever:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
    @classmethod
    def load(cls, path: str | Path) -> "GraphRetriever":
        return cls(load_graph(path))
    def _tokens(self, query: str) -> list[str]:
        seen=set(); out=[]
        for tok in re.findall(r"[a-z0-9][a-z0-9-]*", (query or "").lower()):
            if tok in STOPWORDS: continue
            if len(tok) >= 3 or tok in KEEP:
                if tok not in seen: out.append(tok); seen.add(tok)
        return out
    def retrieve(self, query: str, hops: int = 2, limit: int = 20) -> dict:
        tokens = self._tokens(query)
        seeds=[]
        for node_id, data in self.graph.nodes(data=True):
            hay = f"{node_id} {data.get('name','')}".lower()
            if any(t in hay for t in tokens):
                seeds.append(node_id)
        if not seeds: return {k:list(v) for k,v in EMPTY.items()}
        related=[]; seen=set(); frontier=list(seeds); depth=0
        while frontier and depth <= max(0, hops) and len(related) < limit:
            nxt=[]
            for n in frontier:
                if n in seen: continue
                seen.add(n); related.append(n)
                if len(related) >= limit: break
                neigh = set(self.graph.successors(n)) | set(self.graph.predecessors(n))
                nxt.extend([x for x in neigh if x not in seen])
            frontier=nxt; depth += 1
        node_set=set(related); edge_rows=[]; doc_ids=[]; chunk_ids=[]
        def add_unique(arr, vals):
            for v in vals:
                if v and v not in arr: arr.append(v)
        nodes=[]
        for n in related:
            data=dict(self.graph.nodes[n]); row={"node_id":n, **data}; nodes.append(row)
            add_unique(doc_ids, data.get("doc_ids", [])); add_unique(chunk_ids, data.get("chunk_ids", []))
        for s,t,k,data in self.graph.edges(keys=True, data=True):
            if s in node_set and t in node_set:
                row={"source":s,"target":t,"key":k, **dict(data)}; edge_rows.append(row)
                add_unique(doc_ids, [data.get("doc_id", "")]); add_unique(chunk_ids, [data.get("chunk_id", "")])
        return {"matched_entities": [{"node_id": s, **dict(self.graph.nodes[s])} for s in seeds[:limit]], "related_nodes": nodes, "related_edges": edge_rows, "related_doc_ids": doc_ids, "related_chunk_ids": chunk_ids}
