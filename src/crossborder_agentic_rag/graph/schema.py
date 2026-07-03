"""Lightweight GraphRAG schema objects."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

NODE_TYPES = {"Trademark","Patent","Company","Product","LitigationCase","Classification","LegalConcept"}
EDGE_TYPES = {"REGISTERED_TO","CLASSIFIED_UNDER","ASSERTS_PATENT","ASSERTS_TRADEMARK","SUES","INVOLVES_PRODUCT","CITES_PATENT","CITES_TRADEMARK"}

@dataclass
class GraphNode:
    node_id: str
    node_type: str
    name: str
    source_type: str = ""
    doc_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    def __post_init__(self) -> None:
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"node_type must be one of {sorted(NODE_TYPES)}")
    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "node_type": self.node_type, "name": self.name, "source_type": self.source_type, "doc_ids": list(self.doc_ids), "chunk_ids": list(self.chunk_ids), "metadata": dict(self.metadata)}
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphNode":
        return cls(node_id=str(data["node_id"]), node_type=str(data["node_type"]), name=str(data["name"]), source_type=str(data.get("source_type", "")), doc_ids=list(data.get("doc_ids", [])), chunk_ids=list(data.get("chunk_ids", [])), metadata=dict(data.get("metadata", {})))

@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    doc_id: str = ""
    chunk_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    def __post_init__(self) -> None:
        if self.relation not in EDGE_TYPES:
            raise ValueError(f"relation must be one of {sorted(EDGE_TYPES)}")
    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "relation": self.relation, "doc_id": self.doc_id, "chunk_id": self.chunk_id, "metadata": dict(self.metadata)}
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphEdge":
        return cls(source=str(data["source"]), target=str(data["target"]), relation=str(data["relation"]), doc_id=str(data.get("doc_id", "")), chunk_id=str(data.get("chunk_id", "")), metadata=dict(data.get("metadata", {})))
