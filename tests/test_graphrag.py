from __future__ import annotations
import subprocess, sys
import networkx as nx
from crossborder_agentic_rag.graph.entity_extractor import extract_edges_from_chunk, extract_entities_from_chunk
from crossborder_agentic_rag.graph.graph_builder import build_graph_from_chunks
from crossborder_agentic_rag.graph.graph_retriever import GraphRetriever
from crossborder_agentic_rag.graph.graph_store import load_graph, save_graph
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk

def c(st, md, content="smart luggage patent bag", title="Title", cid="c1", doc="d1"):
    return EvidenceChunk(cid, doc, st, "fixture", title, content, md)

def test_trademark_extraction_gets_mark_and_classification():
    nodes = extract_entities_from_chunk(c("trademark", {"word_mark":"SMARTBAG", "registration_number":"123", "nice_class":"18"}))
    assert {n.node_type for n in nodes} >= {"Trademark", "Classification"}
    assert any(n.node_id == "classification:18" for n in nodes)
    assert any(e.relation == "CLASSIFIED_UNDER" for e in extract_edges_from_chunk(c("trademark", {"word_mark":"SMARTBAG", "nice_class":"18"})))

def test_patent_extraction_gets_patent_and_product():
    nodes = extract_entities_from_chunk(c("patent", {"patent_number":"US123"}, "A smart luggage has GPS lock and battery charging."))
    assert {n.node_type for n in nodes} >= {"Patent", "Product"}
    assert any(n.node_id == "product:smart-luggage" for n in nodes)

def test_litigation_extraction_gets_case_company_patent():
    nodes = extract_entities_from_chunk(c("litigation", {"case_number":"1:23-cv-1", "case_name":"A v B", "name":"Example Corp", "patent_number":"US123"}))
    assert {n.node_type for n in nodes} >= {"LitigationCase", "Company", "Patent"}

def test_build_graph_save_load_and_retrieve(tmp_path):
    chunks = [
        c("patent", {"patent_number":"US123"}, "A smart luggage includes GPS lock.", cid="p1", doc="pd1"),
        c("litigation", {"case_number":"1:23-cv-1", "name":"Example Corp", "patent_number":"US123"}, "Smart luggage litigation", cid="l1", doc="ld1"),
    ]
    graph = build_graph_from_chunks(chunks)
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() >= 3
    paths = save_graph(graph, tmp_path)
    loaded = load_graph(paths["graph"])
    assert loaded.number_of_nodes() == graph.number_of_nodes()
    res = GraphRetriever(loaded).retrieve("smart luggage patent litigation", hops=2, limit=10)
    assert res["related_doc_ids"]
    assert res["related_chunk_ids"]

def test_build_graph_index_help_returns_zero():
    proc = subprocess.run([sys.executable, "scripts/build_graph_index.py", "--help"], text=True, capture_output=True)
    assert proc.returncode == 0
    assert "--chunks-path" in proc.stdout
