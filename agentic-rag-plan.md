# Agentic RAG with Gemma 4 on Azure VM (4 vCPU, 16GB)

**TL;DR:** Deploy a locally-served Gemma 4 E2B (5B params, Q4_K ~3.2GB) via Ollama on an Azure Linux VM. An Agentic RAG pipeline (LangGraph) handles PDF ingestion, hybrid retrieval, re-ranking, and multi-step reasoning. A Streamlit chat UI exposes it to users, backed by a FastAPI server handling both sync and async queries.

---

## Architecture Overview

```
User (Browser)
    │
    ▼
Nginx (port 80)
    ├─► Streamlit UI (:8501) ── chat, upload, source citations
    └─► FastAPI (:8000) ────── /query (sync) + /query/async + /documents/upload
              │
              ▼
        LangGraph Agent
         ├─ Query Router  (simple vs. complex)
         ├─ Query Rewriter (reformulate for retrieval)
         ├─ Hybrid Retriever (ChromaDB vector + BM25 keyword, top-20)
         ├─ Re-ranker (bge-reranker-base, top-5)
         ├─ Context Builder (prompt assembly + metadata)
         └─ Gemma 4 E2B via Ollama (:11434)
```

---

## Phase 1 — Azure VM & Environment Setup

1. Provision **Standard_D4s_v3** (4 vCPU, 16GB, Ubuntu 22.04 LTS) — `$0.19/hr` in eastus
2. Install Docker Engine + Docker Compose v2, Nginx
3. Install Ollama and pull the model:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull gemma4:e2b-q4_k
   ```
4. Verify: `ollama run gemma4:e2b` produces tokens at ~2–4 tok/sec on 4 vCPU

---

## Phase 2 — Project Scaffold

5. Create workspace layout:
   ```
   app/
   app/rag/
   app/api/
   app/ui/
   data/chromadb/
   data/uploads/
   ```
6. `requirements.txt`:
   ```
   langgraph
   langchain-community
   chromadb
   sentence-transformers
   rank_bm25
   fastapi
   uvicorn
   streamlit
   pymupdf
   unstructured[pdf]
   ollama
   ```
7. `app/config.py` — all tunables (chunk size, top-k, model name, Ollama URL)
8. `docker-compose.yml` — services for `ollama`, `app`, `nginx` (*depends_on* ordering)

---

## Phase 3 — PDF Ingestion Pipeline

9. `app/ingest.py`: parse PDFs with PyMuPDF → `unstructured` for tables/headers
10. Semantic chunking: 512 tokens, 50-token overlap, metadata `{filename, page, section, ingested_at}`
11. Embed chunks with **bge-small-en-v1.5** (33M params, CPU-fast, ~90MB) → persist to ChromaDB
12. Build BM25 index (`rank_bm25`) alongside, pickle to disk for fast reload
13. FastAPI `POST /documents/upload` endpoint triggers this pipeline

---

## Phase 4 — LangGraph Agentic RAG Graph

14. `app/rag/graph.py` — define `AgentState` TypedDict:
    ```python
    class AgentState(TypedDict):
        query: str
        rewritten_query: str
        candidates: list
        reranked_docs: list
        answer: str
        sources: list
        route: str  # "simple" | "complex"
    ```
15. `app/rag/nodes.py` — implement each node:

    | Node | Responsibility |
    |---|---|
    | `query_router` | Classify query as `simple` or `complex` |
    | `query_rewriter` | Prompt Gemma to generate 3 search phrasings |
    | `retriever` | ChromaDB cosine top-20 + BM25 top-20 → RRF fusion → top-20 |
    | `reranker` | bge-reranker-base cross-encoder → top-5 |
    | `context_builder` | Assemble final prompt with chunks + metadata citations |
    | `llm_node` | Call Ollama streaming API |
    | `response_formatter` | Return structured `{answer, sources[]}` |

16. Graph edges: `simple` route skips rewriter; `complex` loops through sub-query decomposition up to 3 hops

---

## Phase 5 — FastAPI Backend

17. `app/api/main.py`:

    | Endpoint | Method | Description |
    |---|---|---|
    | `/query` | `POST` | Sync query, simple path, max 30s timeout |
    | `/query/async` | `POST` | Submit to asyncio Queue, returns `{task_id}` |
    | `/query/status/{task_id}` | `GET` | Polling endpoint for async results |
    | `/documents/upload` | `POST` | Multipart PDF upload + ingestion trigger |
    | `/documents/list` | `GET` | List ingested docs from ChromaDB metadata |

---

## Phase 6 — Streamlit Chat UI

18. `app/ui/app.py`:
    - Chat history with `st.session_state`
    - Sidebar: document uploader + list of ingested files
    - Simple queries: stream response from `/query`
    - Complex queries: submit → poll `/query/status/{id}` with progress spinner
    - Source citations displayed below each answer (filename, page)

---

## Phase 7 — Deployment & Hardening

19. `Dockerfile` (multi-stage): install deps, copy app, expose ports 8000 + 8501
20. `docker-compose.yml`: Ollama container with `--num-threads 4`, app container with env vars
21. `nginx.conf`: `location /` → Streamlit; `location /api` → FastAPI; timeouts set to 120s
22. Azure NSG: allow inbound TCP 80, 443, SSH 22; block all else
23. Systemd unit for Docker Compose auto-restart on reboot

---

## Key Files

| File | Purpose |
|---|---|
| `docker-compose.yml` | Orchestrates Ollama + app + Nginx |
| `app/config.py` | Centralized config (chunk size, model, top-k) |
| `app/ingest.py` | PDF parsing → embed → ChromaDB |
| `app/rag/graph.py` | LangGraph state machine definition |
| `app/rag/nodes.py` | All 7 agent node implementations |
| `app/api/main.py` | FastAPI endpoints (sync + async query) |
| `app/ui/app.py` | Streamlit chat UI |
| `requirements.txt` | All Python deps |
| `nginx.conf` | Reverse proxy config |
| `Dockerfile` | Container build |

---

## Verification Checklist

- [ ] `ollama run gemma4:e2b` → tokens stream on terminal
- [ ] Upload 3 internal PDFs via UI → ChromaDB collection count increases
- [ ] Submit simple query → response returned in <30s with page citations
- [ ] Submit complex multi-doc query → task submits, polling shows progress, answer arrives async
- [ ] `docker stats` shows total memory <15GB under concurrent load
- [ ] Nginx serves UI at `http://<VM_IP>` (no port number)

---

## Decisions & Rationale

| Decision | Choice | Reason |
|---|---|---|
| LLM | Gemma 4 E2B Q4_K_M | Model + KV cache fits in ~9GB, leaves ~7GB for stack |
| Vector DB | ChromaDB (embedded) | No server container overhead vs. Qdrant/Weaviate |
| Embeddings | bge-small-en-v1.5 | Best speed/quality on CPU (~100ms/chunk, ~90MB) |
| Orchestration | LangGraph | Lightweight, graph-based state machine, production-grade |
| Async Queue | asyncio-native | No Redis dependency, stays within 16GB memory budget |
| Data scope | Internal PDFs only | Can be extended to live feeds later |

---

## Further Considerations

1. **TLS/HTTPS** — Add Let's Encrypt via Certbot before any production use (`certbot --nginx`)
2. **Authentication** — Add `streamlit-authenticator` or Nginx basic auth for multi-user access
3. **GPU upgrade path** — If 2–4 tok/sec is too slow, swap to `Standard_NC4as_T4_v3` (~$0.53/hr) and add `--n-gpu-layers 35` to Ollama; the rest of the stack is unchanged
