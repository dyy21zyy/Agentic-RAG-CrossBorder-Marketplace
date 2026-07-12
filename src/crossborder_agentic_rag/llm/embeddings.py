from __future__ import annotations
import os
import sys

"""Embedding provider interfaces and implementations."""
import hashlib, math, os
from abc import ABC, abstractmethod

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """FakeEmbeddingProvider is only for tests and smoke runs. It is not semantic."""
    def __init__(self, dim: int = 16) -> None:
        if not isinstance(dim, int): raise TypeError("dim must be an integer")
        if dim <= 0: raise ValueError("dim must be a positive integer")
        self.dim=dim; self.query_calls=[]; self.document_calls=[]
    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text,str): raise TypeError("text must be a string")
        self.query_calls.append(text); return self._embed_text(text)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts,list): raise TypeError("texts must be a list")
        self.document_calls.extend(texts); return [self._embed_text(t) for t in texts]
    def _embed_text(self,text:str)->list[float]:
        if not isinstance(text,str): raise TypeError("text must be a string")
        vals=[]; c=0
        while len(vals)<self.dim:
            for b in hashlib.sha256(f"{text}\0{c}".encode()).digest():
                vals.append(b/127.5-1.0)
                if len(vals)==self.dim: break
            c+=1
        norm=math.sqrt(sum(v*v for v in vals))
        return [0.0 for _ in vals] if norm==0 else [v/norm for v in vals]

class OpenAICompatibleEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str|None=None, api_base: str|None=None, model: str|None=None, dim: int|None=None, timeout: float=60.0)->None:
        self.api_key=api_key or os.getenv("EMBEDDING_API_KEY")
        self.api_base=(api_base or os.getenv("EMBEDDING_API_BASE") or "").rstrip("/")
        self.model=model or os.getenv("EMBEDDING_MODEL")
        env_dim=os.getenv("EMBEDDING_DIM")
        self.dim=dim if dim is not None else (int(env_dim) if env_dim else None)
        self.timeout=timeout
        missing=[n for n,v in {"EMBEDDING_API_KEY":self.api_key,"EMBEDDING_API_BASE":self.api_base,"EMBEDDING_MODEL":self.model}.items() if not v]
        if missing: raise ValueError("Missing embedding API configuration: "+", ".join(missing))
    def embed_query(self,text:str)->list[float]: return self.embed_documents([text])[0]
    def embed_documents(self,texts:list[str])->list[list[float]]:
        import requests
        resp=requests.post(f"{self.api_base}/embeddings",headers={"Authorization":f"Bearer {self.api_key}"},json={"model":self.model,"input":texts},timeout=self.timeout)
        if resp.status_code>=400: raise RuntimeError(f"Embedding API returned status {resp.status_code}: {resp.text[:300]}")
        data=resp.json().get("data",[]); vectors=[]
        for item in data:
            vec=[float(x) for x in item["embedding"]]
            if self.dim is not None and len(vec)!=self.dim: raise ValueError(f"Embedding dimension {len(vec)} does not match expected {self.dim}")
            vectors.append(vec)
        if len(vectors)!=len(texts): raise RuntimeError("Embedding API returned unexpected number of embeddings")
        return vectors

class LocalSentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str|None=None, normalize: bool=True)->None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence-transformers is required for local embeddings. Install with: pip install -e '.[local]'") from exc
        self.model_name=model_name or os.getenv("LOCAL_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
        self.normalize = normalize
        device = os.getenv("EMBEDDING_DEVICE")
        if not device:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self.model = SentenceTransformer(self.model_name, device=device)
        print(f"[embedding] loaded {self.model_name} on {device}", file=sys.stderr)
    def embed_query(self,text:str)->list[float]: return self.embed_documents([text])[0]
    def embed_documents(self,texts:list[str])->list[list[float]]:
        encoded=self.model.encode(texts, normalize_embeddings=self.normalize)
        return [[float(x) for x in vec] for vec in encoded]

def build_embedding_provider(provider: str|None=None, dim: int|None=None)->BaseEmbeddingProvider:
    name=(provider if provider is not None else os.getenv("EMBEDDING_PROVIDER","fake")).lower().replace("_","-")
    if name=="fake": return FakeEmbeddingProvider(dim=16 if dim is None else dim)
    if name in {"openai","openai-compatible","api"}: return OpenAICompatibleEmbeddingProvider(dim=dim)
    if name in {"local","sentence-transformer"}: return LocalSentenceTransformerEmbeddingProvider(os.getenv("LOCAL_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL"))
    raise NotImplementedError(f"Embedding provider '{name}' is not supported.")
