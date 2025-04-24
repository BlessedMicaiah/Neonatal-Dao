from __future__ import annotations
import pathlib
import json
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "knowledge_base"
INDEX_PATH = DATA_DIR / "index.faiss"
META_PATH = DATA_DIR / "index_meta.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents() -> List[str]:
    docs: List[str] = []
    exts = {".txt", ".md"}
    for path in DATA_DIR.rglob("*"):
        if path.suffix.lower() in exts:
            docs.append(path.read_text(encoding="utf-8", errors="ignore"))
    return docs


def build_index() -> None:
    print("[Indexer] Loading documents …")
    documents = load_documents()
    print(f"[Indexer] {len(documents)} documents found")

    print("[Indexer] Initialising embedding model …")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("[Indexer] Encoding documents …")
    embeddings = embedder.encode(documents, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]

    print("[Indexer] Building FAISS index …")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print(f"[Indexer] Saving index to {INDEX_PATH}")
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps({"model": EMBEDDING_MODEL_NAME, "documents": len(documents)}))

    print("[Indexer] Done ✅")


if __name__ == "__main__":
    build_index()
