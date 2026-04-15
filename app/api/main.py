import asyncio
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import OLLAMA_BASE_URL, MODEL_NAME, UPLOADS_PATH
from app.ingest import delete_document, ingest_pdf, list_documents
from app.ingest import get_chroma_collection, get_embedder
from app.rag.graph import run_query
from app.rag.nodes import get_reranker

app = FastAPI(title="Agentic RAG — Business Insights API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup: validate config and warm up models
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    # Validate Ollama is reachable and model exists
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Normalize: ollama list uses "name:tag" format
            configured = MODEL_NAME if ":" in MODEL_NAME else f"{MODEL_NAME}:latest"
            if configured not in models:
                print(
                    f"WARNING: Model '{MODEL_NAME}' not found in Ollama. "
                    f"Available: {models}. Run: ollama pull {MODEL_NAME}"
                )
    except Exception as e:
        print(f"WARNING: Could not reach Ollama at {OLLAMA_BASE_URL}: {e}")

    # Warm up heavy models so first query isn't slow
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_chroma_collection)
    await loop.run_in_executor(None, get_embedder)
    await loop.run_in_executor(None, get_reranker)
    print("Startup complete: models loaded.")

# ---------------------------------------------------------------------------
# In-memory task store (single-worker process — not shared across workers)
# ---------------------------------------------------------------------------
_tasks: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


class AsyncQueryResponse(BaseModel):
    task_id: str


class TaskStatus(BaseModel):
    task_id: str
    status: str          # pending | running | done | error
    answer: Optional[str] = None
    sources: Optional[list] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Query endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query_sync(body: QueryRequest):
    """
    Synchronous query — FastAPI runs sync handlers in a thread pool,
    so the event loop is not blocked.
    Best for simple, single-document questions.
    """
    result = run_query(body.query)
    return QueryResponse(answer=result["answer"], sources=result["sources"])


@app.post("/query/async", response_model=AsyncQueryResponse)
async def query_async(body: QueryRequest, background_tasks: BackgroundTasks):
    """
    Async query — returns a task_id immediately.
    Poll GET /query/status/{task_id} for the result.
    Best for complex multi-document analysis.
    """
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "pending"}
    background_tasks.add_task(_run_graph_background, task_id, body.query)
    return AsyncQueryResponse(task_id=task_id)


@app.get("/query/status/{task_id}", response_model=TaskStatus)
def query_status(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = _tasks[task_id]
    return TaskStatus(task_id=task_id, **task)


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accept a PDF upload, save it, and trigger ingestion."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    os.makedirs(UPLOADS_PATH, exist_ok=True)
    dest = Path(UPLOADS_PATH) / file.filename

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run ingestion in a thread so we don't block the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: ingest_pdf(str(dest), filename=file.filename)
    )

    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["message"])
    if result["status"] == "duplicate":
        raise HTTPException(status_code=409, detail=result["message"])

    return result


@app.get("/documents/list")
def get_documents():
    return list_documents()


@app.delete("/documents/{filename}")
def remove_document(filename: str):
    """Delete all chunks for a document from ChromaDB and BM25 index."""
    result = delete_document(filename)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    # Remove file from uploads folder if it exists
    file_path = Path(UPLOADS_PATH) / filename
    if file_path.exists():
        file_path.unlink()
    return result


# ---------------------------------------------------------------------------
# Background task helper
# ---------------------------------------------------------------------------

async def _run_graph_background(task_id: str, query: str) -> None:
    _tasks[task_id] = {"status": "running"}
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: run_query(query))
        _tasks[task_id] = {
            "status": "done",
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except Exception as exc:
        _tasks[task_id] = {"status": "error", "error": str(exc)}
