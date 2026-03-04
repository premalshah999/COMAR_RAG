"""pipeline/langgraph_pipeline.py — LangGraph agentic RAG pipeline for COMAR.

Graph topology
--------------
::

    START
      │
      ▼
  route_query ──────────────────────────────┐
      │ (no citation in query)              │ (COMAR XX.XX.XX.XX detected)
      ▼                                     ▼
  hybrid_retrieve ◄──────────────── direct_lookup
      │                 (not found)         │ (found)
      └─────────────────────────────────────┘
                        │
                        ▼
                  build_context
                        │
                        ▼
                    generate ──────────────────► hybrid_retrieve  (rewrite loop,
                        │                         max 2 iterations)
                        │ (sufficient OR max iter)
                        ▼
                     verify
                        │
                       END

Usage::

    from pipeline.langgraph_pipeline import run_pipeline

    result = run_pipeline("What permits are needed for pesticide storage?")
    print(result["response"])
    print("Verified citations:", result["verification"]["verified"])
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from pipeline.citation_verifier import CitationVerifier
from pipeline.prompts import QUERY_REWRITE_PROMPT, SYSTEM_PROMPT
from pipeline.router import QueryRouter
from retrieval import COMARRetriever

logger = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────

_CITATION_RE = re.compile(
    r"COMAR\s+\d{1,2}\.\d{2}\.\d{2}(?:\.\d{2})?",
    re.IGNORECASE,
)


class COMARState(TypedDict):
    query: str
    query_type: str
    retrieved_chunks: list[dict]
    rewritten_query: str
    context: str
    response: str
    verification: dict
    iteration_count: int
    # Internal routing flags
    _use_direct_lookup: bool
    _direct_lookup_found: bool
    _needs_rewrite: bool


# ── LLM factory ───────────────────────────────────────────────────────────────

_PLACEHOLDER = "your_key_here"


def _build_llm() -> Any | None:
    """Construct a LangChain chat model from environment variables.

    Priority: deepseek → anthropic → openai → None (stub mode).
    """
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()

    if provider == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY", "")
        model = os.getenv("LLM_MODEL", "deepseek-chat")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if key and key != _PLACEHOLDER:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=model,
                    openai_api_key=key,
                    openai_api_base=base_url,
                )
            except Exception as exc:
                logger.warning("DeepSeek init failed: %s", exc)

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        model = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
        if key and key != _PLACEHOLDER:
            try:
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(model=model, anthropic_api_key=key)
            except Exception as exc:
                logger.warning("Anthropic init failed: %s", exc)

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        if key and key != _PLACEHOLDER:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model=model, openai_api_key=key)
            except Exception as exc:
                logger.warning("OpenAI init failed: %s", exc)

    logger.warning(
        "No valid LLM API key found (LLM_PROVIDER=%s). Running in stub mode.", provider
    )
    return None


# ── Stub generation (no-LLM fallback) ────────────────────────────────────────

def _stub_generate(query: str, context: str) -> str:
    """Generate a response from context without an LLM.

    Extracts real citations from the context and builds a deterministic
    answer that passes citation verification (all cited sections are from
    retrieved context).
    """
    # Extract citations present in the formatted context
    cits = list(dict.fromkeys(_CITATION_RE.findall(context)))[:4]
    cit_refs = "  ".join(f"[{c}]" for c in cits)

    # Pull a brief text excerpt so domain terms (e.g. "dealer") appear
    ctx_start = context.find("[COMAR")
    snippet_raw = context[ctx_start:ctx_start + 500] if ctx_start >= 0 else context[:500]
    # Strip the header line, keep the body text
    lines = [ln for ln in snippet_raw.splitlines() if not ln.startswith("[COMAR") and ln.strip()]
    excerpt = " ".join(lines)[:300].strip()

    if cits:
        response = (
            f"Based on the provided Maryland COMAR regulatory context for the query "
            f'"{query}":\n\n'
            f"{excerpt}\n\n"
            f"The relevant regulatory provisions are: {cit_refs}.\n\n"
            "DISCLAIMER: This information is for research purposes only. "
            "Verify with the Maryland Division of State Documents."
        )
    else:
        response = (
            "The retrieved regulations do not contain enough information to answer "
            "this question definitively. The most relevant section found is in the "
            "retrieved context above. For authoritative guidance, consult the official "
            "COMAR at regs.maryland.gov.\n\n"
            "DISCLAIMER: This information is for research purposes only. "
            "Verify with the Maryland Division of State Documents."
        )
    return response


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph(retriever: COMARRetriever, llm: Any | None) -> Any:
    """Compile and return the LangGraph state machine."""

    router = QueryRouter(llm)
    verifier = CitationVerifier()

    # ── Node implementations ──────────────────────────────────────────────────

    def route_query(state: COMARState) -> dict:
        query_type = router.classify(state["query"])
        use_direct = router.is_citation_lookup(state["query"])
        logger.info("route_query: type=%s direct_lookup=%s", query_type, use_direct)
        return {"query_type": query_type, "_use_direct_lookup": use_direct}

    def direct_lookup(state: COMARState) -> dict:
        query = state["query"]
        # Extract the first COMAR citation string from the query
        match = _CITATION_RE.search(query)
        citation = match.group(0).upper() if match else query

        result = retriever.hybrid.search_by_citation(citation)
        if result:
            logger.info("direct_lookup: found %s", citation)
            return {"retrieved_chunks": [result], "_direct_lookup_found": True}
        logger.info("direct_lookup: not found, falling back to hybrid")
        return {"retrieved_chunks": [], "_direct_lookup_found": False}

    def hybrid_retrieve(state: COMARState) -> dict:
        effective_query = state.get("rewritten_query") or state["query"]
        logger.info("hybrid_retrieve: query=%s", effective_query[:80])
        chunks = retriever.retrieve(effective_query, top_n=8)
        return {"retrieved_chunks": chunks}

    def build_context(state: COMARState) -> dict:
        chunks = state.get("retrieved_chunks", [])
        parts: list[str] = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            citation = chunk.get("citation") or meta.get("citation", "")
            effective_date = meta.get("effective_date") or "N/A"
            context_path = chunk.get("context_path", "")
            chunk_text = chunk.get("chunk_text", "")

            parts.append(
                f"[{citation}] (Effective: {effective_date})\n"
                f"{context_path}\n"
                f"{chunk_text}\n"
                f"---"
            )
        context = "\n\n".join(parts) if parts else "No relevant regulations retrieved."
        return {"context": context}

    def generate(state: COMARState) -> dict:
        query = state["query"]
        context = state.get("context", "")
        iteration = state.get("iteration_count", 0)

        if llm is not None:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
                HumanMessage(content=query),
            ]
            response_obj = llm.invoke(messages)
            response_text = (
                response_obj.content
                if hasattr(response_obj, "content")
                else str(response_obj)
            )
        else:
            response_text = _stub_generate(query, context)

        # Rewrite loop: if insufficient and under iteration limit
        insufficient = "not enough information" in response_text.lower()
        if insufficient and iteration < 2:
            logger.info("generate: insufficient context, rewriting query (iter %d)", iteration)
            retrieved_sections = ", ".join(
                c.get("citation") or c.get("metadata", {}).get("citation", "")
                for c in state.get("retrieved_chunks", [])
            )
            if llm is not None:
                from langchain_core.messages import HumanMessage
                rewrite_prompt = QUERY_REWRITE_PROMPT.format(
                    query=query,
                    retrieved_sections=retrieved_sections,
                )
                rewrite_obj = llm.invoke([HumanMessage(content=rewrite_prompt)])
                rewritten = (
                    rewrite_obj.content.strip()
                    if hasattr(rewrite_obj, "content")
                    else str(rewrite_obj).strip()
                )
            else:
                # Stub rewrite: append "Maryland COMAR regulations" for specificity
                rewritten = f"{query} Maryland COMAR regulations requirements"

            return {
                "response": response_text,
                "rewritten_query": rewritten,
                "iteration_count": iteration + 1,
                "_needs_rewrite": True,
            }

        return {
            "response": response_text,
            "iteration_count": iteration,
            "_needs_rewrite": False,
        }

    def verify(state: COMARState) -> dict:
        response = state.get("response", "")
        chunks = state.get("retrieved_chunks", [])
        verification = verifier.verify(response, chunks)
        final_response = verifier.add_verification_footer(response, verification)
        return {"response": final_response, "verification": verification}

    # ── Conditional edge functions ────────────────────────────────────────────

    def after_route(state: COMARState) -> str:
        return "direct_lookup" if state.get("_use_direct_lookup") else "hybrid_retrieve"

    def after_direct_lookup(state: COMARState) -> str:
        return "build_context" if state.get("_direct_lookup_found") else "hybrid_retrieve"

    def after_generate(state: COMARState) -> str:
        return "hybrid_retrieve" if state.get("_needs_rewrite") else "verify"

    # ── Assemble graph ────────────────────────────────────────────────────────

    g = StateGraph(COMARState)

    g.add_node("route_query", route_query)
    g.add_node("direct_lookup", direct_lookup)
    g.add_node("hybrid_retrieve", hybrid_retrieve)
    g.add_node("build_context", build_context)
    g.add_node("generate", generate)
    g.add_node("verify", verify)

    g.add_edge(START, "route_query")
    g.add_conditional_edges("route_query", after_route)
    g.add_conditional_edges("direct_lookup", after_direct_lookup)
    g.add_edge("hybrid_retrieve", "build_context")
    g.add_edge("build_context", "generate")
    g.add_conditional_edges("generate", after_generate)
    g.add_edge("verify", END)

    return g.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(query: str, llm: Any | None = None) -> dict:
    """Run the full COMAR RAG pipeline for *query*.

    Args:
        query: Natural-language question about Maryland COMAR regulations.
        llm: Optional pre-built LangChain chat model.  When ``None``, the
             model is constructed from environment variables (or stub mode
             is used when no valid API key is present).

    Returns:
        The final :class:`COMARState` dict with keys: ``query``,
        ``query_type``, ``retrieved_chunks``, ``rewritten_query``,
        ``context``, ``response``, ``verification``, ``iteration_count``.
    """
    from dotenv import load_dotenv
    load_dotenv()

    effective_llm = llm if llm is not None else _build_llm()
    retriever = COMARRetriever()
    graph = _build_graph(retriever, effective_llm)

    initial_state: COMARState = {
        "query": query,
        "query_type": "",
        "retrieved_chunks": [],
        "rewritten_query": "",
        "context": "",
        "response": "",
        "verification": {},
        "iteration_count": 0,
        "_use_direct_lookup": False,
        "_direct_lookup_found": False,
        "_needs_rewrite": False,
    }

    result = graph.invoke(initial_state)
    return dict(result)
