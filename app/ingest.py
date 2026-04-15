import os
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from app.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    BM25_INDEX_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# ---------------------------------------------------------------------------
# Lazy singletons — loaded once per process, reused across requests
# ---------------------------------------------------------------------------
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None
_embedder: Optional[SentenceTransformer] = None


def get_chroma_collection():
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-based chunks (~512 tokens each)."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def document_exists(filename: str) -> bool:
    """Return True if this filename is already in the collection."""
    collection = get_chroma_collection()
    results = collection.get(where={"filename": filename}, limit=1, include=[])
    return len(results["ids"]) > 0


def delete_document(filename: str) -> dict:
    """Remove all chunks for a given filename from ChromaDB and rebuild BM25."""
    collection = get_chroma_collection()
    results = collection.get(where={"filename": filename}, include=[])
    ids = results["ids"]
    if not ids:
        return {"status": "not_found", "filename": filename}

    collection.delete(ids=ids)
    _rebuild_bm25_index()
    return {"status": "deleted", "filename": filename, "chunks_removed": len(ids)}


def _rebuild_bm25_index() -> None:
    """Rebuild the BM25 index from whatever remains in ChromaDB."""
    collection = get_chroma_collection()
    result = collection.get(include=["documents"])
    texts: list[str] = result["documents"]
    ids: list[str] = result["ids"]

    index_path = Path(BM25_INDEX_PATH)
    if not texts:
        if index_path.exists():
            index_path.unlink()
        return

    corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(corpus)
    os.makedirs(index_path.parent, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus, "ids": ids}, f)


def ingest_pdf(file_path: str, filename: Optional[str] = None) -> dict:
    """Parse a PDF, chunk, embed, and persist to ChromaDB + BM25 index."""
    if filename is None:
        filename = Path(file_path).name

    if document_exists(filename):
        return {"status": "duplicate", "filename": filename, "message": "Document already ingested."}

    doc = fitz.open(file_path)
    all_chunks: list[dict] = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            continue
        for chunk_idx, chunk in enumerate(chunk_text(text)):
            all_chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "metadata": {
                        "filename": filename,
                        "page": page_num + 1,
                        "chunk_index": chunk_idx,
                        "ingested_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )

    doc.close()

    if not all_chunks:
        return {"status": "error", "message": "No text could be extracted from this PDF."}

    # Embed all chunks in one batched call
    embedder = get_embedder()
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    # Persist to ChromaDB
    collection = get_chroma_collection()
    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in all_chunks],
    )

    # Update on-disk BM25 index
    _update_bm25_index(texts, [c["id"] for c in all_chunks])

    return {
        "status": "success",
        "filename": filename,
        "chunks": len(all_chunks),
    }


# ---------------------------------------------------------------------------
# BM25 index helpers
# ---------------------------------------------------------------------------

def _update_bm25_index(new_texts: list[str], new_ids: list[str]) -> None:
    """Append new documents to the BM25 index and re-persist to disk."""
    index_path = Path(BM25_INDEX_PATH)

    if index_path.exists():
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        corpus: list[list[str]] = data["corpus"]
        ids: list[str] = data["ids"]
    else:
        corpus, ids = [], []

    corpus.extend([t.lower().split() for t in new_texts])
    ids.extend(new_ids)

    bm25 = BM25Okapi(corpus)

    os.makedirs(index_path.parent, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus, "ids": ids}, f)


def load_bm25() -> tuple[Optional[BM25Okapi], list[str]]:
    """Load BM25 index from disk. Returns (None, []) if not yet built."""
    index_path = Path(BM25_INDEX_PATH)
    if not index_path.exists():
        return None, []
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["ids"]


# ---------------------------------------------------------------------------
# Document listing
# ---------------------------------------------------------------------------

def list_documents() -> list[dict]:
    """Return one entry per unique ingested filename with chunk count."""
    collection = get_chroma_collection()
    result = collection.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in result["metadatas"]:
        fname = meta.get("filename", "unknown")
        if fname not in seen:
            seen[fname] = {
                "filename": fname,
                "ingested_at": meta.get("ingested_at", ""),
                "chunks": 0,
            }
        seen[fname]["chunks"] += 1
    return list(seen.values())
