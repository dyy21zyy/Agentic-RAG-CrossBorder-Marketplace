"""Rule-based entity extraction for lightweight GraphRAG."""
from __future__ import annotations
import re
from typing import Any
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
from crossborder_agentic_rag.graph.schema import GraphEdge, GraphNode

PRODUCT_TERMS = ["travel bag", "cooler bag", "smart luggage", "anti-theft", "bag", "luggage", "backpack", "suitcase", "charging", "battery", "gps", "lock"]

def normalize_value(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")

def _first(md: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = md.get(k)
        if v not in (None, "", []):
            return str(v)
    return ""

def _vals(v: Any) -> list[str]:
    if v in (None, ""):
        return []
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v if str(x).strip()]
    return [x.strip() for x in re.split(r"[,;/|]", str(v)) if x.strip()]

def _base(chunk: EvidenceChunk) -> tuple[list[str], list[str]]:
    return ([chunk.doc_id] if chunk.doc_id else [], [chunk.chunk_id] if chunk.chunk_id else [])

def _product_nodes(chunk: EvidenceChunk) -> list[GraphNode]:
    text = f"{chunk.title} {chunk.content}".lower()
    docs, chunks = _base(chunk)
    out = []
    for term in PRODUCT_TERMS:
        if re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", text):
            out.append(GraphNode(f"product:{normalize_value(term)}", "Product", term, chunk.source_type, docs, chunks, {"term": term}))
    return out

def extract_entities_from_chunk(chunk: EvidenceChunk) -> list[GraphNode]:
    md = chunk.metadata or {}; docs, chunks = _base(chunk); nodes: list[GraphNode] = []
    if chunk.source_type == "trademark":
        mark = _first(md, ["word_mark"])
        if mark:
            ident = _first(md, ["registration_number", "serial_number"]) or mark
            nodes.append(GraphNode(f"trademark:{normalize_value(ident)}", "Trademark", mark, "trademark", docs, chunks, {k: md[k] for k in ["registration_number","serial_number","goods_services"] if k in md}))
        for cls in _vals(md.get("nice_classes", md.get("nice_class"))):
            nodes.append(GraphNode(f"classification:{normalize_value(cls)}", "Classification", cls, "trademark", docs, chunks, {"class_value": cls}))
        owner = _first(md, ["owner", "owner_name", "applicant", "company"])
        if owner:
            nodes.append(GraphNode(f"company:{normalize_value(owner)}", "Company", owner, "trademark", docs, chunks, {}))
    elif chunk.source_type == "patent":
        patent = _first(md, ["patent_id", "patent_number"])
        if patent:
            nodes.append(GraphNode(f"patent:{normalize_value(patent)}", "Patent", patent, "patent", docs, chunks, {k: md[k] for k in ["patent_id","patent_number","claim_number"] if k in md}))
        nodes.extend(_product_nodes(chunk))
    elif chunk.source_type == "litigation":
        case = _first(md, ["case_number", "case_row_id"])
        if case:
            nodes.append(GraphNode(f"litigation:{normalize_value(case)}", "LitigationCase", case, "litigation", docs, chunks, {k: md[k] for k in ["case_name","case_row_id","doc_number","short_description"] if k in md}))
        party = _first(md, ["name", "name_long", "party_name", "company"])
        if party:
            nodes.append(GraphNode(f"company:{normalize_value(party)}", "Company", party, "litigation", docs, chunks, {"party_type": md.get("party_type", "")}))
        pat = _first(md, ["patent_number", "patent"])
        if pat:
            nodes.append(GraphNode(f"patent:{normalize_value(pat)}", "Patent", pat, "litigation", docs, chunks, {}))
        tm = _first(md, ["word_mark", "trademark"])
        if tm:
            nodes.append(GraphNode(f"trademark:{normalize_value(tm)}", "Trademark", tm, "litigation", docs, chunks, {}))
        nodes.extend(_product_nodes(chunk))
    return nodes

def extract_edges_from_chunk(chunk: EvidenceChunk) -> list[GraphEdge]:
    nodes = {n.node_type: n for n in extract_entities_from_chunk(chunk)}; md = chunk.metadata or {}; edges: list[GraphEdge] = []
    doc, cid = chunk.doc_id, chunk.chunk_id
    if chunk.source_type == "trademark" and "Trademark" in nodes:
        tm = nodes["Trademark"]
        for n in extract_entities_from_chunk(chunk):
            if n.node_type == "Classification": edges.append(GraphEdge(tm.node_id, n.node_id, "CLASSIFIED_UNDER", doc, cid, {}))
            if n.node_type == "Company": edges.append(GraphEdge(tm.node_id, n.node_id, "REGISTERED_TO", doc, cid, {}))
    elif chunk.source_type == "patent" and "Patent" in nodes:
        for n in extract_entities_from_chunk(chunk):
            if n.node_type == "Product": edges.append(GraphEdge(nodes["Patent"].node_id, n.node_id, "INVOLVES_PRODUCT", doc, cid, {}))
    elif chunk.source_type == "litigation" and "LitigationCase" in nodes:
        case = nodes["LitigationCase"]
        for n in extract_entities_from_chunk(chunk):
            if n.node_type == "Patent": edges.append(GraphEdge(case.node_id, n.node_id, "ASSERTS_PATENT", doc, cid, {}))
            if n.node_type == "Trademark": edges.append(GraphEdge(case.node_id, n.node_id, "ASSERTS_TRADEMARK", doc, cid, {}))
            if n.node_type == "Product": edges.append(GraphEdge(case.node_id, n.node_id, "INVOLVES_PRODUCT", doc, cid, {}))
        # single-row metadata usually cannot identify both sides, so avoid forced SUES edges.
    return edges
