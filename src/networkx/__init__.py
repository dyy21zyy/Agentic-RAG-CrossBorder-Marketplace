"""Tiny NetworkX compatibility shim used when the external dependency is unavailable."""
from __future__ import annotations

class _NodeView:
    def __init__(self, g): self._g=g
    def __call__(self, data=False):
        return list(self._g._nodes.items()) if data else list(self._g._nodes.keys())
    def __getitem__(self, n): return self._g._nodes[n]
    def __iter__(self): return iter(self._g._nodes)

class _EdgeView:
    def __init__(self, g): self._g=g
    def __call__(self, keys=False, data=False):
        rows=[]
        for s, targets in self._g._adj.items():
            for t, keyed in targets.items():
                for k, attrs in keyed.items():
                    if keys and data: rows.append((s,t,k,attrs))
                    elif keys: rows.append((s,t,k))
                    elif data: rows.append((s,t,attrs))
                    else: rows.append((s,t))
        return rows

class MultiDiGraph:
    def __init__(self):
        self._nodes={}; self._adj={}; self._pred={}; self._edge_key=0
        self.nodes=_NodeView(self); self.edges=_EdgeView(self)
    def add_node(self, node, **attrs):
        self._nodes.setdefault(node, {}).update(attrs)
        self._adj.setdefault(node, {}); self._pred.setdefault(node, {})
    def has_node(self, node): return node in self._nodes
    def __contains__(self, node): return node in self._nodes
    def add_edge(self, source, target, key=None, **attrs):
        if source not in self._nodes: self.add_node(source)
        if target not in self._nodes: self.add_node(target)
        if key is None:
            key=self._edge_key; self._edge_key += 1
        self._adj.setdefault(source, {}).setdefault(target, {})[key]=attrs
        self._pred.setdefault(target, {}).setdefault(source, {})[key]=attrs
        return key
    def successors(self, node): return iter(self._adj.get(node, {}).keys())
    def predecessors(self, node): return iter(self._pred.get(node, {}).keys())
    def number_of_nodes(self): return len(self._nodes)
    def number_of_edges(self):
        return sum(len(keys) for targets in self._adj.values() for keys in targets.values())

__all__ = ["MultiDiGraph"]
