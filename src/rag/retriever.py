from __future__ import annotations
import pathlib
from typing import List
import json

import faiss
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "knowledge_base"
INDEX_PATH = DATA_DIR / "index.faiss"
META_PATH = DATA_DIR / "index_meta.json"

class Retriever:
    def __init__(self, k: int = 4):
        self.k = k
        meta = json.loads(META_PATH.read_text())
        self.embedder = SentenceTransformer(meta["model"])
        self.index = faiss.read_index(str(INDEX_PATH))

    def retrieve(self, query: str) -> List[str]:
        from .indexer import load_documents  # lazy import to avoid circular deps
        documents = load_documents()
        q_vec = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(q_vec, self.k)
        return [documents[i] for i in indices[0] if i < len(documents)]
