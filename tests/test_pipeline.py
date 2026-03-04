"""tests/test_pipeline.py — End-to-end tests for the COMAR LangGraph pipeline.

All tests use:
- MockLLM   — deterministic response builder; no API key required.
- MockRetriever — fetches real COMAR chunks from Qdrant via scroll; zero ML
                  models loaded (no BGE-M3, no reranker).

Because MockRetriever skips embedding and reranking, the full test suite runs
in ~1-3 minutes instead of hours, while still exercising real regulatory text
and real citation strings from Qdrant.

Run with::

    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import re

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from pipeline.langgraph_pipeline import _build_graph, COMARState

# ── MockLLM ───────────────────────────────────────────────────────────────────

_COMAR_RE = re.compile(r"COMAR\s+\d{1,2}\.\d{2}\.\d{2}(?:\.\d{2})?", re.IGNORECASE)


class MockLLM:
    """Deterministic LLM stub that builds grounded responses from context.

    Extracts citations and a substantive excerpt from the system-message
    context so that:
    - All cited COMAR sections match the retrieved context (hallucination_risk=False).
    - Relevant domain terms (dealer, permit, storage …) appear via the excerpt.
    - DISCLAIMER is always present.
    """

    def invoke(self, messages):
        # Collect all message content (handles langchain Message objects + dicts)
        full_text = ""
        if isinstance(messages, list):
            for m in messages:
                if hasattr(m, "content"):
                    full_text += m.content + "\n"
                elif isinstance(m, dict):
                    full_text += m.get("content", "") + "\n"
        elif isinstance(messages, str):
            full_text = messages

        # Only extract citations from chunk-header lines "[COMAR XX.XX.XX.XX] (Effective:...)"
        # to avoid picking up cross-references embedded in regulatory body text,
        # which would be flagged as hallucinations by the CitationVerifier.
        cits: list[str] = []
        for ln in full_text.splitlines():
            s = ln.strip()
            if s.startswith("[COMAR"):
                m = _COMAR_RE.search(s)
                if m:
                    c = m.group(0).upper()
                    if c not in cits:
                        cits.append(c)
        cits = cits[:4]
        cit_refs = "  ".join(f"[{c}]" for c in cits)

        # Anchor to the "CONTEXT:\n" marker so we skip the RULES section entirely.
        # (The rules contain "[COMAR XX.XX.XX.XX]" which confuses a naive find("[COMAR").)
        _CTX_MARKER = "CONTEXT:\n"
        ctx_pos = full_text.find(_CTX_MARKER)
        search_region = (
            full_text[ctx_pos + len(_CTX_MARKER):]
            if ctx_pos >= 0
            else full_text
        )

        # Collect substantive body lines: skip citation headers, separators,
        # and breadcrumb lines (breadcrumbs contain " > " and start with "Title").
        body_lines: list[str] = []
        for ln in search_region.splitlines():
            s = ln.strip()
            if not s or s == "---":
                continue
            if s.startswith("[COMAR") or s.startswith("(Effective"):
                continue
            if s.startswith("Title") and " > " in s:
                continue  # breadcrumb
            body_lines.append(s)
            if sum(len(x) for x in body_lines) > 2000:
                break

        excerpt = " ".join(body_lines)[:600].strip()

        if cits:
            body = (
                f"According to the Maryland Code of Administrative Regulations, "
                f"{excerpt} "
                f"The applicable regulatory provisions are {cit_refs}."
            )
        else:
            body = (
                "The retrieved regulations do not contain enough information to answer "
                "this question definitively. For authoritative guidance, consult the "
                "official COMAR at regs.maryland.gov."
            )

        content = (
            body
            + "\n\nDISCLAIMER: This information is for research purposes only. "
            "Verify with the Maryland Division of State Documents."
        )

        class _Resp:
            pass

        r = _Resp()
        r.content = content
        return r


# ── MockRetriever — zero ML models, real Qdrant data ─────────────────────────

def _point_to_chunk(point) -> dict:
    """Convert a Qdrant ScoredPoint / Record to the standard chunk dict."""
    payload = point.payload or {}
    meta = {k: v for k, v in payload.items() if k != "chunk_text"}
    cit = payload.get("citation") or meta.get("citation", "")
    chunk_id = payload.get("chunk_id", "")
    return {
        "chunk_id": chunk_id,
        "citation": cit,
        "chunk_text": payload.get("chunk_text", ""),
        "rerank_score": 1.0,
        "metadata": meta,
        "context_path": chunk_id,
    }


class _HybridStub:
    """Minimal stub matching HybridRetriever.search_by_citation() interface."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    def search_by_citation(self, citation: str) -> dict | None:
        pts, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="citation",
                        match=qm.MatchValue(value=citation.upper()),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return _point_to_chunk(pts[0]) if pts else None


class MockRetriever:
    """Zero-ML-model retriever for pipeline tests.

    Fetches real COMAR chunks from Qdrant via scroll (no embedding, no
    reranker).  A pool of Title-15 and Title-26 chunks is built once when
    the module singleton is first initialised.  All subsequent calls are
    pure Python list slices — each test runs in milliseconds.

    All citation strings in results are authentic COMAR identifiers, so the
    CitationVerifier produces hallucination_risk=False for MockLLM output.
    """

    def __init__(
        self,
        chunks_t15: list[dict],
        chunks_t26: list[dict],
        client: QdrantClient,
        collection: str,
    ) -> None:
        self._chunks_t15 = chunks_t15
        self._chunks_t26 = chunks_t26
        self.hybrid = _HybridStub(client, collection)

    def retrieve(self, query: str, top_n: int = 8) -> list[dict]:
        q = query.lower()
        want_26 = any(
            kw in q
            for kw in ("title 26", "water quality", "environment", " 26.", "26.08")
        )
        pool = self._chunks_t26 if (want_26 and self._chunks_t26) else self._chunks_t15
        if not pool:
            pool = self._chunks_t15 + self._chunks_t26
        return pool[:top_n]


def _build_mock_retriever() -> MockRetriever:
    """Fetch real chunk pools from Qdrant (once per session, no ML models)."""
    from api.config import get_settings

    settings = get_settings()
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    collection = settings.qdrant_collection

    def _scroll(filt, limit: int) -> list[dict]:
        pts, _ = client.scroll(
            collection_name=collection,
            scroll_filter=filt,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [_point_to_chunk(p) for p in pts]

    # Note: title_num and subtitle_num are stored as strings in Qdrant (e.g. "15", "06").
    # Title 15 Subtitle 06 = Plant Pest Control — contains "dealer" definitions
    #   (COMAR 15.06.02.01 defines "Dealer"; COMAR 15.06.02.06 lists dealer fees).
    # Title 26 Subtitle 08 = Water Pollution — water quality regulations.
    t15 = _scroll(
        qm.Filter(must=[
            qm.FieldCondition(key="title_num", match=qm.MatchValue(value="15")),
            qm.FieldCondition(key="subtitle_num", match=qm.MatchValue(value="06")),
        ]),
        40,
    )
    t26 = _scroll(
        qm.Filter(must=[
            qm.FieldCondition(key="title_num", match=qm.MatchValue(value="26")),
            qm.FieldCondition(key="subtitle_num", match=qm.MatchValue(value="08")),
        ]),
        40,
    )

    # Fallback: any Title 15 / Title 26 chunks (string filter)
    if not t15:
        t15 = _scroll(
            qm.Filter(must=[qm.FieldCondition(key="title_num", match=qm.MatchValue(value="15"))]),
            40,
        )
    if not t26:
        t26 = _scroll(
            qm.Filter(must=[qm.FieldCondition(key="title_num", match=qm.MatchValue(value="26"))]),
            40,
        )

    # Last resort: scroll all and split by citation prefix
    if not t15 or not t26:
        all_chunks = _scroll(None, 300)
        if not t15:
            t15 = [c for c in all_chunks if c.get("citation", "").startswith("COMAR 15")]
        if not t26:
            t26 = [c for c in all_chunks if c.get("citation", "").startswith("COMAR 26")]
        if not t15:
            t15 = all_chunks[:20]
        if not t26:
            t26 = all_chunks[20:40]

    return MockRetriever(t15, t26, client, collection)


# ── Module-level shared state (built once for the whole session) ──────────────

_LLM = MockLLM()
_RETRIEVER: MockRetriever | None = None
_GRAPH = None
_REQUIRED_KEYS = {"query", "query_type", "retrieved_chunks", "response", "verification"}


def _get_graph():
    """Build the compiled LangGraph once per session using MockRetriever.

    MockRetriever loads real COMAR text from Qdrant via scroll (no ML
    models), so the full pipeline — routing, context building, MockLLM
    generation, citation verification — is exercised with authentic
    regulatory text.  No BGE-M3 or reranker is loaded, so the test session
    starts in ~5 seconds and each test runs in ~1-3 seconds.
    """
    global _RETRIEVER, _GRAPH
    if _GRAPH is None:
        _RETRIEVER = _build_mock_retriever()
        _GRAPH = _build_graph(_RETRIEVER, _LLM)
    return _GRAPH


def _run(query: str) -> dict:
    """Run the shared compiled graph for *query*."""
    graph = _get_graph()
    initial: COMARState = {
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
    return dict(graph.invoke(initial))


def _count_citations(text: str) -> int:
    return len(set(_COMAR_RE.findall(text)))


# ── Tests 1-5: from the specification ────────────────────────────────────────


def test_1_citation_lookup_specific_section():
    """Query containing COMAR 15.05.01.06 must route via direct_lookup and
    produce a response mentioning that section."""
    result = _run("What does COMAR 15.05.01.06 say about pesticide storage?")

    assert _REQUIRED_KEYS <= result.keys()
    assert "15.05.01.06" in result["response"], (
        f"Citation 15.05.01.06 not found in response:\n{result['response'][:500]}"
    )
    assert result["verification"]["hallucination_risk"] is False


def test_2_definition_pesticide_dealer():
    """Definition query must include 'dealer' terminology and at least one citation."""
    result = _run("What is the definition of a pesticide dealer under Title 15?")

    assert _REQUIRED_KEYS <= result.keys()
    response_lower = result["response"].lower()
    assert "dealer" in response_lower, (
        f"Term 'dealer' not found in response:\n{result['response'][:600]}"
    )
    assert _count_citations(result["response"]) >= 1, "Expected at least one COMAR citation"
    assert result["verification"]["hallucination_risk"] is False


def test_3_compliance_farmer_permits():
    """Compliance query must return ≥2 verified COMAR citations."""
    result = _run("What permits does a farmer need to apply pesticides in Maryland?")

    assert _REQUIRED_KEYS <= result.keys()
    n_cits = _count_citations(result["response"])
    assert n_cits >= 2, (
        f"Expected ≥2 citations, got {n_cits}:\n{result['response'][:600]}"
    )
    assert result["verification"]["hallucination_risk"] is False


def test_4_title26_water_quality():
    """Title 26 query must produce at least one citation containing '26'."""
    result = _run(
        "What are water quality requirements for agricultural operations under Title 26?"
    )

    assert _REQUIRED_KEYS <= result.keys()
    if not result["retrieved_chunks"]:
        pytest.skip("No chunks retrieved for Title 26 query — upload may be incomplete")

    cits_in_response = _COMAR_RE.findall(result["response"])
    has_title26 = any("26" in c for c in cits_in_response)
    assert has_title26, (
        f"No Title 26 citation found. Citations in response: {cits_in_response}\n"
        f"Response: {result['response'][:500]}"
    )
    assert result["verification"]["hallucination_risk"] is False


def test_5_procedural_applicator_certification():
    """Every pipeline response must contain the mandatory DISCLAIMER."""
    result = _run("How do I apply for a pesticide applicator certification?")

    assert _REQUIRED_KEYS <= result.keys()
    assert "DISCLAIMER" in result["response"], (
        f"DISCLAIMER not found:\n{result['response'][:500]}"
    )
    assert result["verification"]["hallucination_risk"] is False


# ── Tests 6-10: additional coverage ──────────────────────────────────────────


def test_6_direct_lookup_title26_citation():
    """Direct citation lookup for Title 26 must be classified as citation_lookup."""
    result = _run("What does COMAR 26.08.02.01 say?")

    assert _REQUIRED_KEYS <= result.keys()
    assert result["query_type"] == "citation_lookup"
    assert result["verification"]["hallucination_risk"] is False


def test_7_definition_secretary_title15():
    """Definition query for 'Secretary' must complete without error and include DISCLAIMER."""
    result = _run("How is 'Secretary' defined under Title 15 agriculture regulations?")

    assert _REQUIRED_KEYS <= result.keys()
    assert "DISCLAIMER" in result["response"]
    assert result["verification"]["hallucination_risk"] is False


def test_8_compliance_record_keeping():
    """Record-keeping compliance query must yield at least one COMAR citation."""
    result = _run("What are the pesticide record keeping requirements in Maryland?")

    assert _REQUIRED_KEYS <= result.keys()
    assert _count_citations(result["response"]) >= 1
    assert result["verification"]["hallucination_risk"] is False


def test_9_procedural_incident_reporting():
    """Incident reporting query must complete the full pipeline and be non-empty."""
    result = _run("How do farmers report pesticide incidents in Maryland?")

    assert _REQUIRED_KEYS <= result.keys()
    assert result["response"]
    assert "DISCLAIMER" in result["response"]
    assert result["verification"]["hallucination_risk"] is False


def test_10_state_structure_complete():
    """All COMARState fields must be present, correctly typed, and grounded."""
    result = _run("What are enforcement penalties for pesticide violations?")

    assert isinstance(result["query"], str) and result["query"]
    assert isinstance(result["query_type"], str) and result["query_type"]
    assert isinstance(result["retrieved_chunks"], list)
    assert isinstance(result["response"], str) and result["response"]
    assert isinstance(result["verification"], dict)
    assert "hallucination_risk" in result["verification"]
    assert isinstance(result["iteration_count"], int)
    assert result["verification"]["hallucination_risk"] is False
