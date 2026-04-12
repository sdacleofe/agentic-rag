"""
Agentic RAG node implementations.

Each node is a pure function: (state: dict) -> dict (updated state).
Nodes are composed into a LangGraph StateGraph in graph.py.
"""

from typing import Optional

import httpx
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import (
    OLLAMA_BASE_URL,
    MODEL_NAME,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    NUM_PREDICT,
    NUM_CTX,
    NUM_THREAD,
    TEMPERATURE,
)
from app.ingest import get_chroma_collection, get_embedder, load_bm25

# ---------------------------------------------------------------------------
# Lazy singleton — reranker is large (~350MB), load once
# ---------------------------------------------------------------------------
_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ---------------------------------------------------------------------------
# Node: query_router
# ---------------------------------------------------------------------------
_COMPLEX_KEYWORDS = {
    "compare", "contrast", "analyze", "analyse", "trend", "over time",
    "relationship", "impact", "multiple", "across", "summarize all",
    "why", "how does", "how do", "explain", "reasons", "factors",
    "what caused", "what are the", "implications",
}


def query_router(state: dict) -> dict:
    """Classify query as 'simple' or 'complex' using lightweight heuristics."""
    query: str = state["query"]
    words = query.lower().split()
    is_complex = len(words) > 15 or any(kw in query.lower() for kw in _COMPLEX_KEYWORDS)
    return {**state, "route": "complex" if is_complex else "simple"}


# ---------------------------------------------------------------------------
# Node: query_rewriter
# ---------------------------------------------------------------------------

def query_rewriter(state: dict) -> dict:
    """Generate 3 alternative search phrasings via Gemma to improve recall."""
    query: str = state["query"]
    prompt = (
        "Generate 3 concise and distinct search queries to retrieve relevant "
        "business and economic information for the following question.\n"
        "Rules: one query per line, no numbering, no explanation, no empty lines.\n\n"
        f"Question: {query}\n\nQueries:"
    )
    raw = _call_ollama(prompt, max_tokens=120)
    rewrites = [line.strip() for line in raw.strip().splitlines() if line.strip()][:3]
    # Always keep the original query first
    return {**state, "rewritten_queries": [query] + rewrites}


# ---------------------------------------------------------------------------
# Node: retriever
# ---------------------------------------------------------------------------

def retriever(state: dict) -> dict:
    """
    Hybrid retrieval: ChromaDB vector search + BM25 keyword search.
    Results are merged via Reciprocal Rank Fusion (RRF, k=60).
    """
    queries: list[str] = state.get("rewritten_queries") or [state["query"]]
    collection = get_chroma_collection()
    embedder = get_embedder()
    bm25, bm25_ids = load_bm25()

    doc_count = collection.count()
    if doc_count == 0:
        return {**state, "candidates": []}

    n_results = min(RETRIEVAL_TOP_K, doc_count)
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for q in queries:
        # --- Vector search ---
        q_emb = embedder.encode(q).tolist()
        vec = collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            include=["documents", "metadatas"],
        )
        for rank, (doc_id, text, meta) in enumerate(
            zip(vec["ids"][0], vec["documents"][0], vec["metadatas"][0])
        ):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + 60)
            doc_map.setdefault(doc_id, {"id": doc_id, "text": text, "metadata": meta})

        # --- BM25 search ---
        if bm25 is not None and bm25_ids:
            tokens = q.lower().split()
            scores = bm25.get_scores(tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
                :RETRIEVAL_TOP_K
            ]
            missing_ids: list[str] = []
            for rank, idx in enumerate(top_indices):
                if idx >= len(bm25_ids):
                    continue
                doc_id = bm25_ids[idx]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + 60)
                if doc_id not in doc_map:
                    missing_ids.append(doc_id)

            # Batch-fetch any BM25 hits not already in doc_map
            if missing_ids:
                fetched = collection.get(
                    ids=missing_ids, include=["documents", "metadatas"]
                )
                for fid, ftext, fmeta in zip(
                    fetched["ids"], fetched["documents"], fetched["metadatas"]
                ):
                    doc_map[fid] = {"id": fid, "text": ftext, "metadata": fmeta}

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:RETRIEVAL_TOP_K]
    candidates = [doc_map[did] for did in sorted_ids if did in doc_map]
    return {**state, "candidates": candidates}


# ---------------------------------------------------------------------------
# Node: reranker
# ---------------------------------------------------------------------------

def reranker_node(state: dict) -> dict:
    """Re-rank candidate chunks with a cross-encoder; keep top RERANK_TOP_K."""
    query: str = state["query"]
    candidates: list[dict] = state["candidates"]

    if not candidates:
        return {**state, "reranked_docs": []}

    cross_encoder = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores: list[float] = cross_encoder.predict(pairs).tolist()

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:RERANK_TOP_K]
    return {**state, "reranked_docs": [doc for _, doc in ranked]}


# ---------------------------------------------------------------------------
# Node: context_builder
# ---------------------------------------------------------------------------

def context_builder(state: dict) -> dict:
    """Assemble the final LLM prompt from top-ranked chunks."""
    query: str = state["query"]
    docs: list[dict] = state["reranked_docs"]

    if not docs:
        prompt = (
            "You are a business and economic analyst assistant.\n"
            "No relevant documents were found in the knowledge base for this question.\n"
            "Politely explain that you cannot answer without relevant source material "
            "and suggest the user uploads relevant documents.\n\n"
            f"Question: {query}\n\nResponse:"
        )
        return {**state, "prompt": prompt, "sources": []}

    context_parts: list[str] = []
    sources: list[dict] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc["metadata"]
        ref = f"[{i}] {meta.get('filename', 'unknown')} — page {meta.get('page', '?')}"
        context_parts.append(f"{ref}\n{doc['text']}")
        sources.append(
            {"ref": i, "filename": meta.get("filename", "unknown"), "page": meta.get("page")}
        )

    context = "\n\n---\n\n".join(context_parts)
    prompt = (
        "You are a business and economic analyst assistant. "
        "Answer the question using ONLY the provided context. "
        "Cite every claim using [N] notation matching the source headers above.\n"
        "If the context does not contain enough information, say so explicitly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return {**state, "prompt": prompt, "sources": sources}


# ---------------------------------------------------------------------------
# Node: llm_node
# ---------------------------------------------------------------------------

def llm_node(state: dict) -> dict:
    """Send the assembled prompt to Ollama and retrieve the generated answer."""
    answer = _call_ollama(state["prompt"], max_tokens=NUM_PREDICT)
    return {**state, "answer": answer}


# ---------------------------------------------------------------------------
# Node: response_formatter (pass-through — state already has answer + sources)
# ---------------------------------------------------------------------------

def response_formatter(state: dict) -> dict:
    return state


# ---------------------------------------------------------------------------
# Internal: Ollama HTTP call
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, max_tokens: int = NUM_PREDICT) -> str:
    """Blocking call to the Ollama /api/generate endpoint."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_thread": NUM_THREAD,
        },
    }
    with httpx.Client(timeout=180.0) as client:
        resp = client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()
    return resp.json().get("response", "")
