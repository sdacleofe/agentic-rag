"""
Streamlit chat UI for the Business & Economic Insights RAG system.

All API calls are server-side (httpx sync).
The API_BASE env var points to the FastAPI service (Docker network or localhost).
"""

import os
import time

import httpx
import streamlit as st

API_BASE: str = os.getenv("API_BASE", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Business Insights RAG",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Business & Economic Insights")
st.caption("Powered by Gemma 4 · Agentic RAG · Internal Documents")

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
                st.markdown(f"- `{d['filename']}`")
        else:
            st.caption("No documents uploaded yet.")
    except Exception:
        st.caption("⚠️ Could not reach API.")

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

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
