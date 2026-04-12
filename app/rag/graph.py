"""
LangGraph state machine for Agentic RAG.

Graph flow:
  router ──► (simple) ──────────────────► retriever
         └─► (complex) ──► rewriter ──► retriever
                                             │
                                         reranker
                                             │
                                       context_builder
                                             │
                                           llm
                                             │
                                         formatter
                                             │
                                           END
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END

from app.rag.nodes import (
    context_builder,
    llm_node,
    query_rewriter,
    query_router,
    reranker_node,
    response_formatter,
    retriever,
)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    route: str                  # "simple" | "complex"
    rewritten_queries: list     # [original] + up to 3 rewrites
    candidates: list            # raw retrieval hits (dicts)
    reranked_docs: list         # cross-encoder ranked hits (dicts)
    prompt: str                 # assembled LLM prompt
    answer: str                 # generated answer
    sources: list               # [{ref, filename, page}, ...]


# ---------------------------------------------------------------------------
# Conditional edge: route after query_router
# ---------------------------------------------------------------------------

def _route_query(state: AgentState) -> str:
    return "rewrite" if state.get("route") == "complex" else "retrieve"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router", query_router)
    g.add_node("rewriter", query_rewriter)
    g.add_node("retriever", retriever)
    g.add_node("reranker", reranker_node)
    g.add_node("context_builder", context_builder)
    g.add_node("llm", llm_node)
    g.add_node("formatter", response_formatter)

    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        _route_query,
        {"rewrite": "rewriter", "retrieve": "retriever"},
    )
    g.add_edge("rewriter", "retriever")
    g.add_edge("retriever", "reranker")
    g.add_edge("reranker", "context_builder")
    g.add_edge("context_builder", "llm")
    g.add_edge("llm", "formatter")
    g.add_edge("formatter", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Singleton compiled graph — one instance per process
# ---------------------------------------------------------------------------

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _initial_state(query: str) -> AgentState:
    return AgentState(
        query=query,
        route="",
        rewritten_queries=[],
        candidates=[],
        reranked_docs=[],
        prompt="",
        answer="",
        sources=[],
    )


def run_query(query: str) -> dict:
    """Convenience wrapper: invoke the graph and return answer + sources."""
    result = get_graph().invoke(_initial_state(query))
    return {"answer": result["answer"], "sources": result["sources"]}
