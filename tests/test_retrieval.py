"""tests/test_retrieval.py — Integration tests for the COMAR retrieval stack.

All tests require:
  - Qdrant running on localhost:6333 with the comar_regulations collection
    populated (at least Title 15 fully uploaded).
  - ./data/comar_graph.pkl present.
  - PYTHONPATH=. when invoked from the project root.

Run with::

    pytest tests/test_retrieval.py -v
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from ingestion.embedder import Embedder
from retrieval import COMARRetriever
from retrieval.graph_expander import GraphExpander
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

# ── Shared fixtures ───────────────────────────────────────────────────────────

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION = "comar_regulations"


@pytest.fixture(scope="module")
def qdrant_client() -> QdrantClient:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
    info = client.get_collection(COLLECTION)
    if (info.points_count or 0) < 100:
        pytest.skip("Qdrant collection not sufficiently populated")
    return client


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


@pytest.fixture(scope="module")
def hybrid(qdrant_client: QdrantClient, embedder: Embedder) -> HybridRetriever:
    return HybridRetriever(qdrant_client, COLLECTION, embedder)


@pytest.fixture(scope="module")
def expander() -> GraphExpander:
    return GraphExpander()


@pytest.fixture(scope="module")
def reranker() -> Reranker:
    return Reranker()


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_hybrid_returns_results(hybrid: HybridRetriever) -> None:
    """Hybrid retrieval on a pesticide query must return ≥5 results."""
    results = hybrid.retrieve("pesticide storage", top_k=10)

    assert len(results) >= 5, f"Expected ≥5 results, got {len(results)}"

    # Every result must have the required keys
    required_keys = {"chunk_id", "chunk_text", "rrf_score", "metadata"}
    for r in results:
        missing = required_keys - r.keys()
        assert not missing, f"Result missing keys: {missing}"

    # Scores must be positive and in descending order
    scores = [r["rrf_score"] for r in results]
    assert all(s > 0 for s in scores), "All RRF scores must be positive"
    assert scores == sorted(scores, reverse=True), "Results must be sorted by rrf_score"


def test_citation_lookup(hybrid: HybridRetriever) -> None:
    """Exact citation lookup must return the correct COMAR regulation."""
    result = hybrid.search_by_citation("COMAR 15.05.01.06")

    assert result is not None, "Expected a result for citation COMAR 15.05.01.06"
    assert result["metadata"].get("citation") == "COMAR 15.05.01.06", (
        f"Citation mismatch: {result['metadata'].get('citation')}"
    )
    assert result["chunk_text"], "chunk_text must not be empty"


def test_graph_expansion_adds_definitions(expander: GraphExpander) -> None:
    """Graph expansion must add the .01 definition chunk for a chapter."""
    graph = expander.graph

    # Dynamically find a chapter that has:
    # (a) a definition regulation (.01 chunk_type)
    # (b) at least one other non-definition regulation
    target_def_id: str | None = None
    target_reg_id: str | None = None

    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") != "regulation":
            continue
        if data.get("chunk_type") != "definition":
            continue

        # Found a definition node — look for a sibling in the same chapter
        from ingestion.graph_builder import get_chapter_for_regulation

        chapter_id = get_chapter_for_regulation(node_id, graph)
        if chapter_id is None:
            continue

        # Find a sibling regulation (non-definition) in this chapter
        for _, sibling, edge_data in graph.out_edges(chapter_id, data=True):
            if (
                edge_data.get("edge_type") == "CONTAINS"
                and sibling != node_id
                and graph.nodes[sibling].get("chunk_type") != "definition"
            ):
                target_def_id = node_id
                target_reg_id = sibling
                break

        if target_def_id:
            break

    assert target_def_id is not None, (
        "Could not find a chapter with both a definition and a non-definition regulation"
    )
    assert target_reg_id is not None

    # Expand the non-definition regulation — expect the .01 definition to appear
    expanded = expander.expand([target_reg_id])

    assert target_def_id in expanded, (
        f"Expected definition chunk '{target_def_id}' in expanded ids {expanded}"
    )


def test_reranker_changes_order(reranker: Reranker) -> None:
    """Reranker must reorder candidates based on cross-encoder relevance."""
    query = "Maryland pesticide applicator license requirements"

    # Build 10 candidates — put the highly relevant one at the end (rank 10)
    candidates = [
        {
            "chunk_id": f"FAKE.00.00.00.{i:02d}",
            "chunk_text": f"This regulation concerns unrelated topic number {i}.",
            "rrf_score": 1.0 / (i + 1),
            "metadata": {},
        }
        for i in range(9)
    ]
    # Highly relevant candidate appended last
    candidates.append(
        {
            "chunk_id": "FAKE.15.05.01.10",
            "chunk_text": (
                "A person must hold a valid pesticide applicator license issued by the "
                "Maryland Department of Agriculture before applying pesticides commercially "
                "in the State of Maryland.  License requirements include passing an exam "
                "and completing continuing education credits."
            ),
            "rrf_score": 0.001,  # deliberately low rrf_score (ranked last)
            "metadata": {},
        }
    )

    reranked = reranker.rerank(query, candidates, top_n=5)

    assert len(reranked) == 5, f"Expected 5 results, got {len(reranked)}"

    # Ensure rerank_score key is present on all results
    for r in reranked:
        assert "rerank_score" in r, "rerank_score key missing from result"

    # The highly relevant candidate should be ranked #1 by the cross-encoder
    top_chunk_id = reranked[0]["chunk_id"]
    assert top_chunk_id == "FAKE.15.05.01.10", (
        f"Expected relevant candidate at rank 1, got '{top_chunk_id}'"
    )


def test_full_retrieve_pipeline() -> None:
    """End-to-end pipeline must return ≥3 results with all required keys."""
    retriever = COMARRetriever()

    results = retriever.retrieve(
        "What permits are required for pesticide application in Maryland?",
        top_n=8,
    )

    assert len(results) >= 3, f"Expected ≥3 results, got {len(results)}"

    required_keys = {"chunk_id", "citation", "chunk_text", "rerank_score", "metadata", "context_path"}
    for r in results:
        missing = required_keys - r.keys()
        assert not missing, f"Result missing keys: {missing}"
        assert r["chunk_text"], "chunk_text must not be empty"
        assert r["context_path"], "context_path must not be empty"
        assert isinstance(r["rerank_score"], float)
