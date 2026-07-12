"""Build NetworkX graphs from evidence chunks."""
from __future__ import annotations
import logging
import networkx as nx
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
from crossborder_agentic_rag.graph.entity_extractor import extract_edges_from_chunk, extract_entities_from_chunk

LOGGER = logging.getLogger(__name__)

def _merge_unique(base, extra):
    """
    Fast unique merge for full GraphRAG build.

    The old version repeatedly rebuilt set(base) for high-frequency entities,
    which becomes very slow on full chunk corpus. This version keeps behavior
    unchanged but avoids unnecessary work for empty inputs.
    """
    if not extra:
        return list(base or [])
    if not base:
        out = []
        seen = set()
    else:
        out = list(base)
        seen = set(out)

    for v in extra:
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out

def _merge_metadata(old: dict, new: dict) -> dict:
    merged = dict(old or {})
    for k, v in (new or {}).items():
        if k not in merged or merged[k] in (None, "", []):
            merged[k] = v
    return merged

def build_graph_from_chunks(chunks: list[EvidenceChunk]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for chunk in chunks:
        try:
            nodes = extract_entities_from_chunk(chunk)
            edges = extract_edges_from_chunk(chunk)
        except Exception as exc:
            LOGGER.warning("Skipping chunk %s during graph extraction: %s", getattr(chunk, "chunk_id", "<unknown>"), exc)
            continue
        for node in nodes:
            data = node.to_dict(); node_id = data.pop("node_id")
            if graph.has_node(node_id):
                cur = graph.nodes[node_id]
                cur["doc_ids"] = _merge_unique(list(cur.get("doc_ids", [])), node.doc_ids)
                cur["chunk_ids"] = _merge_unique(list(cur.get("chunk_ids", [])), node.chunk_ids)
                cur["metadata"] = _merge_metadata(cur.get("metadata", {}), node.metadata)
                if not cur.get("source_type") and node.source_type: cur["source_type"] = node.source_type
            else:
                graph.add_node(node_id, **data)
        for edge in edges:
            if edge.source in graph and edge.target in graph:
                graph.add_edge(edge.source, edge.target, relation=edge.relation, doc_id=edge.doc_id, chunk_id=edge.chunk_id, metadata=dict(edge.metadata))
    return graph
