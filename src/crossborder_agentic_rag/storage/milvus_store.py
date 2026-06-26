"""Real Milvus vector storage adapter."""
from __future__ import annotations
import json
from typing import Any
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk
REQUIRED_FILTER_FIELDS={"source_type","source_subtype","doc_id","case_row_id","case_number","patent_id","patent_number","registration_number","word_mark"}
META_FIELDS=["case_row_id","case_number","patent_id","patent_number","registration_number","word_mark"]
def _load_pymilvus():
    try: import pymilvus
    except ImportError as exc: raise ImportError("pymilvus is required for real Milvus mode. Install with: pip install -e '.[milvus]'") from exc
    return pymilvus
def _esc(s:str)->str: return s.replace('\\','\\\\').replace('"','\\"')
class MilvusChunkStore:
    def __init__(self,uri:str,token:str|None,collection_name:str,embedding_dim:int,overwrite:bool=False,metric_type:str="COSINE",index_type:str="HNSW",index_params:dict|None=None,search_params:dict|None=None)->None:
        self.uri=uri; self.token=token; self.collection_name=collection_name; self.embedding_dim=embedding_dim; self.overwrite=overwrite; self.metric_type=metric_type; self.index_type=index_type; self.index_params=index_params or {"M":16,"efConstruction":200}; self.search_params=search_params or {"ef":64}; self.collection=None
    def connect(self)->None:
        pm=_load_pymilvus(); pm.connections.connect(alias="default",uri=self.uri,token=self.token)
    def build_filter_expr(self,filters:dict|None=None,source_types:list[str]|None=None)->str:
        parts=[]
        merged=dict(filters or {})
        if source_types: merged["source_type"]=source_types
        for k,v in merged.items():
            if k not in REQUIRED_FILTER_FIELDS: raise ValueError(f"Unsupported Milvus filter field: {k}")
            if isinstance(v,str): parts.append(f'{k} == "{_esc(v)}"')
            elif isinstance(v,list) and all(isinstance(x,str) for x in v): parts.append(f'{k} in [{", ".join(chr(34)+_esc(x)+chr(34) for x in v)}]')
            else: raise ValueError(f"Unsupported filter value for {k}: {v!r}")
        return " and ".join(parts)
    def ensure_collection(self)->None:
        pm=_load_pymilvus();
        if self.overwrite and pm.utility.has_collection(self.collection_name): pm.utility.drop_collection(self.collection_name)
        if not pm.utility.has_collection(self.collection_name):
            fields=[pm.FieldSchema(name="chunk_id",dtype=pm.DataType.VARCHAR,is_primary=True,max_length=256),pm.FieldSchema(name="doc_id",dtype=pm.DataType.VARCHAR,max_length=256),pm.FieldSchema(name="source_type",dtype=pm.DataType.VARCHAR,max_length=64),pm.FieldSchema(name="source_subtype",dtype=pm.DataType.VARCHAR,max_length=128),pm.FieldSchema(name="title",dtype=pm.DataType.VARCHAR,max_length=1024),pm.FieldSchema(name="content",dtype=pm.DataType.VARCHAR,max_length=65535),pm.FieldSchema(name="dense_vector",dtype=pm.DataType.FLOAT_VECTOR,dim=self.embedding_dim),pm.FieldSchema(name="metadata_json",dtype=pm.DataType.VARCHAR,max_length=65535)]
            fields += [pm.FieldSchema(name=f,dtype=pm.DataType.VARCHAR,max_length=512) for f in META_FIELDS]
            schema=pm.CollectionSchema(fields=fields,description="IP evidence chunks")
            self.collection=pm.Collection(name=self.collection_name,schema=schema)
        else: self.collection=pm.Collection(self.collection_name)
    def create_indexes(self)->None:
        if self.collection is None: self.ensure_collection()
        self.collection.create_index("dense_vector", {"index_type":self.index_type,"metric_type":self.metric_type,"params":self.index_params})
        self.collection.load()
    def _row(self,c:EvidenceChunk,v:list[float])->dict[str,Any]:
        return {"chunk_id":c.chunk_id,"doc_id":c.doc_id,"source_type":c.source_type,"source_subtype":c.source_subtype,"title":c.title,"content":c.content,"dense_vector":[float(x) for x in v],"metadata_json":json.dumps(c.metadata,ensure_ascii=False),**{f:str(c.metadata.get(f,"")) for f in META_FIELDS}}
    def insert_chunks(self,chunks:list[EvidenceChunk],vectors:list[list[float]])->int:
        if len(chunks)!=len(vectors): raise ValueError("chunks and vectors must have the same length")
        if not chunks: return 0
        for v in vectors:
            if len(v)!=self.embedding_dim: raise ValueError(f"Vector dimension {len(v)} does not match expected {self.embedding_dim}")
        if self.collection is None: self.ensure_collection()
        self.collection.insert([self._row(c,v) for c,v in zip(chunks,vectors)])
        return len(chunks)
    def dense_search(self,vector:list[float],filters:dict|None=None,top_k:int=20)->list[EvidenceChunk]:
        if len(vector)!=self.embedding_dim: raise ValueError(f"Vector dimension {len(vector)} does not match expected {self.embedding_dim}")
        if top_k<=0: return []
        if self.collection is None: self.ensure_collection()
        res=self.collection.search(data=[vector],anns_field="dense_vector",param={"metric_type":self.metric_type,"params":self.search_params},limit=top_k,expr=self.build_filter_expr(filters),output_fields=["chunk_id","doc_id","source_type","source_subtype","title","content","metadata_json"])
        hits=res[0] if res else []; out=[]
        for h in hits:
            ent=getattr(h,"entity",{})
            get=ent.get if hasattr(ent,"get") else lambda k: getattr(ent,k)
            md=json.loads(get("metadata_json") or "{}")
            out.append(EvidenceChunk(get("chunk_id"),get("doc_id"),get("source_type"),get("source_subtype"),get("title"),get("content"),md,float(getattr(h,"distance",getattr(h,"score",0.0)))))
        return out
    def drop_collection(self)->None:
        pm=_load_pymilvus();
        if pm.utility.has_collection(self.collection_name): pm.utility.drop_collection(self.collection_name)
    def flush(self)->None:
        if self.collection is not None: self.collection.flush()
