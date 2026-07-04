"""Real Milvus vector storage adapter."""
from __future__ import annotations
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Any
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk

REQUIRED_FILTER_FIELDS={"source_type","source_subtype","doc_id","case_row_id","case_number","patent_id","patent_number","registration_number","word_mark"}
META_FIELDS=["case_row_id","case_number","patent_id","patent_number","registration_number","word_mark"]
PARTITION_BY_SOURCE={"trademark":"trademark_db","patent":"patent_db","litigation":"litigation_db"}

MISSING_PYMILVUS_MESSAGE = "pymilvus is required for Milvus retrieval. Install pymilvus or use bm25_only mode."

def _load_pymilvus():
    """Import pymilvus without letting a local .db MILVUS_URI break import-time validation."""
    old_uri = os.environ.get("MILVUS_URI")
    removed = False
    if old_uri and _is_lite_uri(old_uri):
        removed = True
        os.environ.pop("MILVUS_URI", None)
    try:
        import pymilvus
    except Exception as exc:
        raise ImportError(MISSING_PYMILVUS_MESSAGE) from exc
    finally:
        if removed:
            os.environ["MILVUS_URI"] = old_uri  # restore for the application
    if pymilvus is None:
        raise ImportError(MISSING_PYMILVUS_MESSAGE)
    return pymilvus

def _is_lite_uri(uri: str) -> bool:
    u=(uri or "").lower(); return u.endswith(".db") or u.startswith("file:")

def _normalize_lite_uri(uri: str) -> str:
    if uri.lower().startswith("file:"):
        uri = uri[5:]
    return str(Path(uri).expanduser().resolve()) if _is_lite_uri(uri) else uri

VARCHAR_MAX_BYTES={"chunk_id":256,"doc_id":256,"source_type":64,"source_subtype":128,"title":1024,"content":65535,"parent_id":256,"context_path":2048,"partition":64,"metadata_json":65535,**{f:512 for f in META_FIELDS}}

def _varchar(value: Any, max_bytes: int) -> str:
    text = "" if value is None else str(value)
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore")

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

def partition_for_source(source_type:str)->str:
    if source_type not in PARTITION_BY_SOURCE: raise ValueError(f"Unsupported source_type for Milvus partition: {source_type!r}")
    return PARTITION_BY_SOURCE[source_type]

class MilvusChunkStore:
    def __init__(self,uri:str,token:str|None,collection_name:str,embedding_dim:int,overwrite:bool=False,metric_type:str="COSINE",index_type:str="HNSW",index_params:dict|None=None,search_params:dict|None=None)->None:
        self.uri=_normalize_lite_uri(uri); self.token=token; self.collection_name=collection_name; self.embedding_dim=embedding_dim; self.overwrite=overwrite; self.metric_type=metric_type; self.index_type=index_type; self.index_params=index_params or {"M":16,"efConstruction":200}; self.search_params=search_params or {"ef":64}; self.collection=None; self.client=None; self.is_lite=_is_lite_uri(self.uri); self._overwrite_done=False
    def connect(self)->None:
        pm=_load_pymilvus()
        if self.is_lite:
            Path(self.uri).expanduser().parent.mkdir(parents=True, exist_ok=True)
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
        if self.is_lite:
            if self.client is None: self.connect()
            has_collection = getattr(self.client, "has_collection", None)
            if has_collection is None:
                self._overwrite_done=True
                return
            if self.overwrite and not self._overwrite_done and has_collection(self.collection_name):
                self.client.drop_collection(collection_name=self.collection_name)
            self._overwrite_done=True
            if not has_collection(self.collection_name):
                schema=self.client.create_schema(auto_id=False, enable_dynamic_field=False)
                for name,max_len in VARCHAR_MAX_BYTES.items():
                    schema.add_field(field_name=name, datatype=pm.DataType.VARCHAR, is_primary=(name=="chunk_id"), max_length=max_len)
                schema.add_field(field_name="dense_vector", datatype=pm.DataType.FLOAT_VECTOR, dim=self.embedding_dim)
                index_params=self.client.prepare_index_params()
                index_params.add_index(field_name="dense_vector", index_type="FLAT", metric_type=self.metric_type)
                self.client.create_collection(collection_name=self.collection_name, schema=schema, index_params=index_params)
            return
        if self.overwrite and not self._overwrite_done and pm.utility.has_collection(self.collection_name): pm.utility.drop_collection(self.collection_name)
        self._overwrite_done=True
        if not pm.utility.has_collection(self.collection_name):
            fields=[pm.FieldSchema(name="chunk_id",dtype=pm.DataType.VARCHAR,is_primary=True,max_length=256),pm.FieldSchema(name="doc_id",dtype=pm.DataType.VARCHAR,max_length=256),pm.FieldSchema(name="source_type",dtype=pm.DataType.VARCHAR,max_length=64),pm.FieldSchema(name="source_subtype",dtype=pm.DataType.VARCHAR,max_length=128),pm.FieldSchema(name="title",dtype=pm.DataType.VARCHAR,max_length=1024),pm.FieldSchema(name="content",dtype=pm.DataType.VARCHAR,max_length=65535),pm.FieldSchema(name="parent_id",dtype=pm.DataType.VARCHAR,max_length=256),pm.FieldSchema(name="context_path",dtype=pm.DataType.VARCHAR,max_length=2048),pm.FieldSchema(name="partition",dtype=pm.DataType.VARCHAR,max_length=64),pm.FieldSchema(name="dense_vector",dtype=pm.DataType.FLOAT_VECTOR,dim=self.embedding_dim),pm.FieldSchema(name="metadata_json",dtype=pm.DataType.VARCHAR,max_length=65535)]
            fields += [pm.FieldSchema(name=f,dtype=pm.DataType.VARCHAR,max_length=512) for f in META_FIELDS]
            schema=pm.CollectionSchema(fields=fields,description="IP evidence chunks")
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
        parent_id=str(c.metadata.get("parent_id",c.doc_id))
        context_path=str(c.metadata.get("context_path",""))
        partition=str(c.metadata.get("partition",PARTITION_BY_SOURCE.get(c.source_type,"")))
        metadata_json=json.dumps(c.metadata,ensure_ascii=False)
        row={"chunk_id":c.chunk_id,"doc_id":c.doc_id,"source_type":c.source_type,"source_subtype":c.source_subtype,"title":c.title,"content":c.content,"parent_id":parent_id,"context_path":context_path,"partition":partition,"dense_vector":[float(x) for x in v],"metadata_json":metadata_json,**{f:c.metadata.get(f,"") for f in META_FIELDS}}
        for field,max_bytes in VARCHAR_MAX_BYTES.items(): row[field]=_varchar(row.get(field,""), max_bytes)
        return row
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
            if self.client is None: self.connect()
            if getattr(self.client, "has_collection", lambda name: True)(self.collection_name):
                self.client.drop_collection(collection_name=self.collection_name)
        elif pm.utility.has_collection(self.collection_name): pm.utility.drop_collection(self.collection_name)
    def flush(self)->None:
        if self.is_lite:
            if self.client is not None:
                try: self.client.flush(collection_name=self.collection_name)
                except TypeError: self.client.flush([self.collection_name])
            return
        if self.collection is not None: self.collection.flush()
    def close(self)->None:
        if self.client is not None:
            try: self.client.close()
            except Exception: pass
            self.client=None
    def row_count(self)->int|None:
        try:
            self.flush()
            if self.is_lite:
                self.load_collection()
                stats=self.client.get_collection_stats(collection_name=self.collection_name)
                return int(stats.get("row_count", stats.get("row_count ", 0)))
            if self.collection is None: self.ensure_collection()
            self.collection.flush(); return int(getattr(self.collection,"num_entities"))
        except Exception:
            try:
                res=self.client.query(collection_name=self.collection_name, filter="", output_fields=["count(*)"]) if self.is_lite else None
                if res: return int(res[0].get("count(*)", res[0].get("count", 0)))
            except Exception: pass
        return None
