"""Lightweight GraphRAG retriever over a NetworkX graph."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import networkx as nx

from crossborder_agentic_rag.graph.graph_store import load_graph


STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "what", "when", "where", "which", "about", "into", "over", "under", "between",
    "against", "use", "using", "useful", "please", "show", "find",
}

# 这些是领域泛词，不适合作为 graph seed。
GENERIC_GRAPH_TOKENS = {
    "patent", "patents",
    "trademark", "trademarks",
    "litigation", "case", "cases",
    "company", "companies",
    "plaintiff", "defendant",
    "infringement", "risk", "network",
    "connected", "connect", "connection",
    "entity", "entities",
    "evidence", "graph", "relationship", "relationships",
}

EMPTY = {
    "matched_entities": [],
    "related_nodes": [],
    "related_edges": [],
    "related_doc_ids": [],
    "related_chunk_ids": [],
}


class GraphRetriever:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    @classmethod
    def load(cls, path: str | Path) -> "GraphRetriever":
        return cls(load_graph(path))

    def _tokens(self, query: str) -> list[str]:
        seen = set()
        out = []

        for tok in re.findall(r"[a-z0-9][a-z0-9-]*", (query or "").lower()):
            if tok in STOPWORDS:
                continue
            if tok in GENERIC_GRAPH_TOKENS:
                continue
            if len(tok) < 3:
                continue
            if tok not in seen:
                out.append(tok)
                seen.add(tok)

        return out

    def _prefix(self, node_id: str) -> str:
        return str(node_id).split(":", 1)[0].lower() if ":" in str(node_id) else ""

    def _neighbors(self, node_id: str) -> set[str]:
        out = set()
        try:
            out |= set(self.graph.successors(node_id))
        except Exception:
            pass
        try:
            out |= set(self.graph.predecessors(node_id))
        except Exception:
            pass
        return out

    def _degree(self, node_id: str) -> int:
        return len(self._neighbors(node_id))

    def _is_noisy_prefix(self, node_id: str) -> bool:
        return self._prefix(node_id) in {"classification"}

    def _node_score(self, node_id: str, data: dict[str, Any], tokens: list[str], raw_query: str) -> int:
        node_l = str(node_id).lower()
        name_l = str(data.get("name", "")).lower().strip()

        # classification 这类 hub 节点默认不能作为 seed。
        if self._is_noisy_prefix(node_l):
            return 0

        score = 0

        # 精确 node_id 匹配最强，例如 product:battery / patent:12185715。
        if node_l == raw_query.strip():
            score += 1000
        elif len(node_l) >= 8 and node_l in raw_query:
            score += 300

        # 短 name 不允许 substring 匹配，避免 A / ION / DEF 误命中。
        if name_l and len(name_l) >= 4:
            name_tokens = set(re.findall(r"[a-z0-9][a-z0-9-]*", name_l))

            if name_l == raw_query.strip():
                score += 500

            # name 的完整 token 与 query token 相交。
            overlap = name_tokens & set(tokens)
            if overlap:
                score += 80 * len(overlap)

        # token 可以匹配 node_id，但要求 token 至少 4 位。
        for t in tokens:
            if len(t) >= 4 and t in node_l:
                score += 30

        return score

    def retrieve(
        self,
        query: str,
        hops: int = 2,
        limit: int = 50,
        seed_limit: int = 10,
        edge_limit: int = 100,
        chunk_limit: int = 500,
        max_expand_degree: int = 800,
    ) -> dict:
        raw_query = (query or "").lower().strip()
        tokens = self._tokens(query)

        if not raw_query:
            return {k: list(v) for k, v in EMPTY.items()}

        candidates = []

        for node_id, data in self.graph.nodes(data=True):
            score = self._node_score(str(node_id), dict(data), tokens, raw_query)
            if score <= 0:
                continue

            deg = self._degree(str(node_id))
            candidates.append((score, deg, str(node_id)))

        if not candidates:
            return {k: list(v) for k, v in EMPTY.items()}

        # 分数优先，度数次之。限制 seed 数量，避免 seed 占满 limit。
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        seeds = [node_id for _, _, node_id in candidates[: max(1, seed_limit)]]

        related = []
        seen = set()
        frontier = list(seeds)
        depth = 0

        while frontier and depth <= max(0, hops) and len(related) < limit:
            nxt = []

            for n in frontier:
                if n in seen:
                    continue

                seen.add(n)
                related.append(n)

                if len(related) >= limit:
                    break

                # classification hub 不继续扩展。
                if self._is_noisy_prefix(n):
                    continue

                # 非 seed 的超高 degree 节点不继续扩展，避免爆炸。
                if n not in seeds and self._degree(n) > max_expand_degree:
                    continue

                for x in self._neighbors(n):
                    if x not in seen:
                        nxt.append(x)

            frontier = nxt
            depth += 1

        node_set = set(related)
        nodes = []
        edge_rows = []
        doc_ids = []
        chunk_ids = []

        def add_unique(arr, vals, max_len=None):
            for v in vals or []:
                if not v:
                    continue
                if v not in arr:
                    arr.append(v)
                    if max_len is not None and len(arr) >= max_len:
                        return

        for n in related:
            data = dict(self.graph.nodes[n])
            row = {"node_id": n, **data}
            nodes.append(row)
            add_unique(doc_ids, data.get("doc_ids", []), chunk_limit)
            add_unique(chunk_ids, data.get("chunk_ids", []), chunk_limit)

        seen_edges = set()

        for s, t, k, data in self.graph.edges(keys=True, data=True):
            if s not in node_set or t not in node_set:
                continue

            data = dict(data)
            rel_type = data.get("type") or data.get("edge_type") or data.get("relation") or "related_to"
            chunk_id = data.get("chunk_id", "")

            # 对重复边去重。没有 chunk_id 的边，按 source-target-relation 去重。
            dedupe_key = (s, t, rel_type, chunk_id or "")
            if dedupe_key in seen_edges:
                continue
            seen_edges.add(dedupe_key)

            edge_rows.append({"source": s, "target": t, "key": k, **data})
            add_unique(doc_ids, [data.get("doc_id", "")], chunk_limit)
            add_unique(chunk_ids, [chunk_id], chunk_limit)

            if len(edge_rows) >= edge_limit:
                break

        return {
            "matched_entities": [
                {"node_id": s, **dict(self.graph.nodes[s])}
                for s in seeds[:limit]
            ],
            "related_nodes": nodes,
            "related_edges": edge_rows,
            "related_doc_ids": doc_ids[:chunk_limit],
            "related_chunk_ids": chunk_ids[:chunk_limit],
        }
