"""
Streamlit chat UI for the Business & Economic Insights RAG system.

All API calls are server-side (httpx sync).
The API_BASE env var points to the FastAPI service (Docker network or localhost).
"""

import json
import os
import time
from pathlib import Path

import httpx
import streamlit as st

API_BASE: str = os.getenv("API_BASE", "http://localhost:8000")
CHAT_HISTORY_FILE = Path("/data/chat_history.json")


# ---------------------------------------------------------------------------
# Chat persistence helpers
# ---------------------------------------------------------------------------

def _load_history() -> list:
    try:
        if CHAT_HISTORY_FILE.exists():
            return json.loads(CHAT_HISTORY_FILE.read_text())
    except Exception:
        pass
    return []


def _save_history(messages: list) -> None:
    try:
        CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHAT_HISTORY_FILE.write_text(json.dumps(messages))
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Business Insights RAG",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Business & Economic Insights")
st.caption("Powered by Llama 3.2 · Agentic RAG · Internal Documents")

# ---------------------------------------------------------------------------
# Sidebar — document management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Knowledge Base")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        with st.spinner(f"Ingesting **{uploaded.name}** …"):
            try:
                resp = httpx.post(
                    f"{API_BASE}/documents/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=300.0,
                )
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Ingested **{data['filename']}** — {data['chunks']} chunks")
            except httpx.HTTPStatusError as e:
                st.error(f"Upload failed: {e.response.text}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.divider()
    st.subheader("Ingested Files")
    try:
        docs = httpx.get(f"{API_BASE}/documents/list", timeout=10.0).json()
        if docs:
            for d in docs:
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"- `{d['filename']}` ({d.get('chunks', '?')} chunks)")
                if col2.button("🗑️", key=f"del_{d['filename']}", help=f"Delete {d['filename']}"):
                    try:
                        r = httpx.delete(f"{API_BASE}/documents/{d['filename']}", timeout=30.0)
                        r.raise_for_status()
                        st.success(f"Deleted **{d['filename']}**")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
        else:
            st.caption("No documents uploaded yet.")
    except Exception:
        st.caption("⚠️ Could not reach API.")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = []
        _save_history([])
        st.rerun()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = _load_history()

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for s in msg["sources"]:
                    st.markdown(
                        f"**[{s['ref']}]** {s['filename']} — page {s.get('page', '?')}"
                    )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
_COMPLEX_KEYWORDS = {
    "compare", "contrast", "analyze", "analyse", "trend", "over time",
    "relationship", "impact", "multiple", "across", "summarize",
    "why", "how does", "how do", "explain", "reasons", "factors",
    "implications", "what caused", "what are the",
}

if prompt := st.chat_input("Ask a business or economic question …"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    _save_history(st.session_state["messages"])
    with st.chat_message("user"):
        st.markdown(prompt)

    words = prompt.lower().split()
    is_complex = len(words) > 15 or any(k in prompt.lower() for k in _COMPLEX_KEYWORDS)

    with st.chat_message("assistant"):
        answer = ""
        sources: list = []

        if not is_complex:
            # ----------------------------------------------------------------
            # Synchronous path — simple queries
            # ----------------------------------------------------------------
            with st.spinner("Thinking …"):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/query",
                        json={"query": prompt},
                        timeout=120.0,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                except httpx.HTTPStatusError as e:
                    answer = f"Error from API: {e.response.text}"
                except Exception as e:
                    answer = f"Request failed: {e}"
        else:
            # ----------------------------------------------------------------
            # Asynchronous path — complex / multi-document queries
            # ----------------------------------------------------------------
            with st.spinner("Submitting deep analysis …"):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/query/async",
                        json={"query": prompt},
                        timeout=30.0,
                    )
                    resp.raise_for_status()
                    task_id: str = resp.json()["task_id"]
                except Exception as e:
                    st.error(f"Failed to submit analysis: {e}")
                    st.stop()

            progress = st.progress(0, text="Analysing documents …")
            timed_out = True
            for tick in range(120):
                time.sleep(3)
                try:
                    status_resp = httpx.get(
                        f"{API_BASE}/query/status/{task_id}", timeout=10.0
                    )
                    task = status_resp.json()
                except Exception:
                    continue

                pct = min((tick + 1) / 80, 0.95)
                progress.progress(pct, text=f"Status: {task['status']} …")

                if task["status"] == "done":
                    progress.progress(1.0, text="Complete ✓")
                    answer = task["answer"]
                    sources = task.get("sources", [])
                    timed_out = False
                    break
                elif task["status"] == "error":
                    answer = f"Analysis error: {task.get('error', 'unknown error')}"
                    timed_out = False
                    break

            if timed_out:
                answer = "Analysis timed out (>6 min). Try a more focused question."

        st.markdown(answer)
        if sources:
            with st.expander("Sources", expanded=True):
                for s in sources:
                    st.markdown(
                        f"**[{s['ref']}]** {s['filename']} — page {s.get('page', '?')}"
                    )

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    _save_history(st.session_state["messages"])
