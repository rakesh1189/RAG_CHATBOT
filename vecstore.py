import os
from typing import List, Tuple, Dict, Any
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_dir: str):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
        self._collections: Dict[str, Any] = {}

    def get_or_create(self, doc_id: str):
        if doc_id not in self._collections:
            self._collections[doc_id] = self.client.get_or_create_collection(name=doc_id, metadata={"hnsw:space": "cosine"})
        return self._collections[doc_id]

    def add(self, doc_id: str, ids: List[str], embeddings: List[List[float]], metadatas: List[dict], documents: List[str]):
        col = self.get_or_create(doc_id)
        col.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self, doc_id: str, query_embedding: List[float], top_k: int = 8):
        col = self.get_or_create(doc_id)
        res = col.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances"])
        return res
