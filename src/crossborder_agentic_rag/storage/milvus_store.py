"""Real Milvus vector storage adapter."""
from __future__ import annotations
import json
from collections import defaultdict
from typing import Any
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk

REQUIRED_FILTER_FIELDS={"source_type","source_subtype","doc_id","case_row_id","case_number","patent_id","patent_number","registration_number","word_mark"}
META_FIELDS=["case_row_id","case_number","patent_id","patent_number","registration_number","word_mark"]
PARTITION_BY_SOURCE={"trademark":"trademark_db","patent":"patent_db","litigation":"litigation_db"}

MISSING_PYMILVUS_MESSAGE = "pymilvus is required for Milvus retrieval. Install pymilvus or use bm25_only mode."

def _load_pymilvus():
    # PyMilvus parses MILVUS_URI at import time and expects an http(s) URI.
    # For Milvus Lite, this project passes a local .db path explicitly to
    # MilvusClient(uri=...). Hide MILVUS_URI only during import.
    import os

    saved_uri = os.environ.pop("MILVUS_URI", None)
    try:
        import pymilvus
    except Exception as exc:
        raise ImportError(MISSING_PYMILVUS_MESSAGE) from exc
    finally:
        if saved_uri is not None:
            os.environ["MILVUS_URI"] = saved_uri

    if pymilvus is None:
        raise ImportError(MISSING_PYMILVUS_MESSAGE)
    return pymilvus

def _is_lite_uri(uri: str) -> bool:
    u=(uri or "").lower(); return u.endswith(".db") or u.startswith("file:")

def parse_metadata_json(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {"_metadata_parse_error": True}
    return parsed if isinstance(parsed, dict) else {"_metadata_parse_error": True}

def _esc(s:str)->str: return s.replace('\\','\\\\').replace('"','\\"')

def _varchar(value: Any, max_len: int = 512) -> str:
    """Convert value to a safe Milvus VARCHAR string."""
    if value is None:
        return ""
    s = str(value)
    b = s.encode("utf-8")
    if len(b) <= max_len:
        return s
    return b[:max_len].decode("utf-8", errors="ignore")

def partition_for_source(source_type:str)->str:
    if source_type not in PARTITION_BY_SOURCE: raise ValueError(f"Unsupported source_type for Milvus partition: {source_type!r}")
    return PARTITION_BY_SOURCE[source_type]

class MilvusChunkStore:
    def __init__(self,uri:str,token:str|None,collection_name:str,embedding_dim:int,overwrite:bool=False,metric_type:str="COSINE",index_type:str="HNSW",index_params:dict|None=None,search_params:dict|None=None)->None:
        self.uri=uri; self.token=token; self.collection_name=collection_name; self.embedding_dim=embedding_dim; self.overwrite=overwrite; self.metric_type=metric_type; self.index_type=index_type; self.index_params=index_params or {"M":16,"efConstruction":200}; self.search_params=search_params or {"ef":64}; self.collection=None; self.client=None; self.is_lite=_is_lite_uri(uri)
    def connect(self)->None:
        pm=_load_pymilvus()
        if self.is_lite:
            self.client=pm.MilvusClient(uri=self.uri, token=self.token) if self.token else pm.MilvusClient(uri=self.uri)
        else:
            pm.connections.connect(alias="default",uri=self.uri,token=self.token)
    def build_filter_expr(self,filters:dict|None=None,source_types:list[str]|None=None)->str:
        parts=[]; merged=dict(filters or {})
        if source_types: merged["source_type"]=source_types
        for k,v in merged.items():
            if k not in REQUIRED_FILTER_FIELDS: raise ValueError(f"Unsupported Milvus filter field: {k}")
            if isinstance(v,str): parts.append(f'{k} == "{_esc(v)}"')
            elif isinstance(v,list) and all(isinstance(x,str) for x in v): parts.append(f'{k} in [{", ".join(chr(34)+_esc(x)+chr(34) for x in v)}]')
            else: raise ValueError(f"Unsupported filter value for {k}: {v!r}")
        return " and ".join(parts)
    def ensure_collection(self)->None:
        pm=_load_pymilvus()
        fields=[pm.FieldSchema(name="chunk_id",dtype=pm.DataType.VARCHAR,is_primary=True,max_length=256),pm.FieldSchema(name="doc_id",dtype=pm.DataType.VARCHAR,max_length=256),pm.FieldSchema(name="source_type",dtype=pm.DataType.VARCHAR,max_length=64),pm.FieldSchema(name="source_subtype",dtype=pm.DataType.VARCHAR,max_length=128),pm.FieldSchema(name="title",dtype=pm.DataType.VARCHAR,max_length=1024),pm.FieldSchema(name="content",dtype=pm.DataType.VARCHAR,max_length=65535),pm.FieldSchema(name="parent_id",dtype=pm.DataType.VARCHAR,max_length=256),pm.FieldSchema(name="context_path",dtype=pm.DataType.VARCHAR,max_length=2048),pm.FieldSchema(name="partition",dtype=pm.DataType.VARCHAR,max_length=64),pm.FieldSchema(name="dense_vector",dtype=pm.DataType.FLOAT_VECTOR,dim=self.embedding_dim),pm.FieldSchema(name="metadata_json",dtype=pm.DataType.VARCHAR,max_length=65535)]
        fields += [pm.FieldSchema(name=f,dtype=pm.DataType.VARCHAR,max_length=512) for f in META_FIELDS]
        schema=pm.CollectionSchema(fields=fields,description="IP evidence chunks")

        if self.is_lite:
            if self.client is None:
                self.connect()

            # In Milvus Lite, insert_chunks() calls ensure_collection() for every batch.
            # If overwrite stays True, each batch drops the previous batch.
            # Therefore overwrite must be consumed only once.
            if self.overwrite:
                if self.client.has_collection(collection_name=self.collection_name):
                    self.client.drop_collection(collection_name=self.collection_name)
                self.overwrite = False

            if not self.client.has_collection(collection_name=self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                )
            return

        if self.overwrite and pm.utility.has_collection(self.collection_name): pm.utility.drop_collection(self.collection_name)
        if not pm.utility.has_collection(self.collection_name):
            self.collection=pm.Collection(name=self.collection_name,schema=schema)
        else: self.collection=pm.Collection(self.collection_name)
        self.ensure_partitions()
    def ensure_partitions(self)->None:
        if self.is_lite: return
        if self.collection is None:
            self.ensure_collection(); return
        existing={p.name for p in getattr(self.collection,"partitions",[]) or []}
        for part in PARTITION_BY_SOURCE.values():
            if part not in existing: self.collection.create_partition(part)
    def load_collection(self)->None:
        self.ensure_collection()
        try:
            if self.is_lite:
                self.client.load_collection(collection_name=self.collection_name)
            elif self.collection is not None:
                self.collection.load()
        except Exception as exc:
            msg=str(exc).lower()
            if "loaded" in msg or "already" in msg:
                return
            raise
    def create_indexes(self)->None:
        if self.collection is None: self.ensure_collection()
        if self.is_lite: return
        self.collection.create_index("dense_vector", {"index_type":self.index_type,"metric_type":self.metric_type,"params":self.index_params})
        self.load_collection()
    def _row(self,c:EvidenceChunk,v:list[float])->dict[str,Any]:
        parent_id=c.metadata.get("parent_id",c.doc_id)
        context_path=c.metadata.get("context_path","")
        partition=c.metadata.get("partition",PARTITION_BY_SOURCE.get(c.source_type,""))
        metadata_json=json.dumps(c.metadata,ensure_ascii=False)
        return {
            "chunk_id":_varchar(c.chunk_id,256),
            "doc_id":_varchar(c.doc_id,256),
            "source_type":_varchar(c.source_type,64),
            "source_subtype":_varchar(c.source_subtype,128),
            "title":_varchar(c.title,1024),
            "content":_varchar(c.content,65535),
            "parent_id":_varchar(parent_id,256),
            "context_path":_varchar(context_path,2048),
            "partition":_varchar(partition,64),
            "dense_vector":[float(x) for x in v],
            "metadata_json":_varchar(metadata_json,65535),
            **{f:_varchar(c.metadata.get(f,""),512) for f in META_FIELDS},
        }

    def insert_chunks(self,chunks:list[EvidenceChunk],vectors:list[list[float]])->int:
        if len(chunks)!=len(vectors): raise ValueError("chunks and vectors must have the same length")
        if not chunks: return 0
        for v in vectors:
            if len(v)!=self.embedding_dim: raise ValueError(f"Vector dimension {len(v)} does not match expected {self.embedding_dim}")
        if self.collection is None and not self.is_lite: self.ensure_collection()
        rows=[self._row(c,v) for c,v in zip(chunks,vectors)]
        if self.is_lite:
            self.ensure_collection(); self.client.insert(collection_name=self.collection_name, data=rows)
        else:
            self.ensure_partitions()
            groups=defaultdict(list)
            for c,row in zip(chunks,rows): groups[partition_for_source(c.source_type)].append(row)
            for part,part_rows in groups.items(): self.collection.insert(part_rows, partition_name=part)
        return len(chunks)
    def _hit_to_chunk(self,h:Any)->EvidenceChunk:
        ent=getattr(h,"entity",None) or (h.get("entity",{}) if isinstance(h,dict) else {})
        def get(k, default=""):
            if hasattr(ent,"get"): return ent.get(k, default)
            return getattr(ent,k,default)
        distance = h.get("distance", h.get("score", 0.0)) if isinstance(h,dict) else getattr(h,"distance",getattr(h,"score",0.0))
        return EvidenceChunk(str(get("chunk_id")),str(get("doc_id")),str(get("source_type")),str(get("source_subtype")),str(get("title")),str(get("content")),parse_metadata_json(get("metadata_json")),float(distance))
    def dense_search(self,vector:list[float],filters:dict|None=None,source_types:list[str]|None=None,top_k:int=20)->list[EvidenceChunk]:
        if len(vector)!=self.embedding_dim: raise ValueError(f"Vector dimension {len(vector)} does not match expected {self.embedding_dim}")
        if top_k<=0: return []
        self.load_collection()
        expr=self.build_filter_expr(filters,source_types=source_types)
        fields=["chunk_id","doc_id","source_type","source_subtype","title","content","metadata_json"]
        partition_names=[PARTITION_BY_SOURCE[st] for st in (source_types or []) if st in PARTITION_BY_SOURCE]
        if self.is_lite:
            res=self.client.search(collection_name=self.collection_name,data=[vector],anns_field="dense_vector",search_params={"metric_type":self.metric_type,"params":self.search_params},limit=top_k,filter=expr or None,output_fields=fields)
        else:
            res=self.collection.search(data=[vector],anns_field="dense_vector",param={"metric_type":self.metric_type,"params":self.search_params},limit=top_k,expr=expr,output_fields=fields,partition_names=partition_names or None)
        return [self._hit_to_chunk(h) for h in (res[0] if res else [])]
    def drop_collection(self)->None:
        pm=_load_pymilvus()
        if self.is_lite:
            self.ensure_collection(); self.client.drop_collection(collection_name=self.collection_name)
        elif pm.utility.has_collection(self.collection_name): pm.utility.drop_collection(self.collection_name)
    def flush(self)->None:
        if self.is_lite: return
        if self.collection is not None: self.collection.flush()
