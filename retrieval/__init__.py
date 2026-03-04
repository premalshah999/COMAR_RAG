"""retrieval/__init__.py — Unified COMAR retrieval pipeline.

Orchestrates three stages:

1. **Hybrid retrieval** — dense + sparse vector search (top-100) via
   :class:`~retrieval.hybrid_retriever.HybridRetriever`.
2. **Graph expansion** — adds cross-referenced regulations and chapter
   definition chunks via :class:`~retrieval.graph_expander.GraphExpander`.
3. **Reranking** — cross-encoder scoring to select the best *top_n* results
   via :class:`~retrieval.reranker.Reranker`.

Public interface::

    from retrieval import COMARRetriever

    retriever = COMARRetriever()
    results   = retriever.retrieve("What permits are required for pesticide application?")
    for r in results:
        print(r["citation"], r["rerank_score"])
        print(r["context_path"])
        print(r["chunk_text"][:200])
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from api.config import get_settings
from ingestion.embedder import Embedder
from retrieval.graph_expander import GraphExpander
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

__all__ = ["COMARRetriever"]


class COMARRetriever:
    """End-to-end retrieval pipeline for COMAR regulations.

    Initialises all sub-components from environment configuration on
    construction.  The BGE-M3 embedder and reranker models are loaded lazily
    on their first use.

    Args:
        top_k_hybrid: Number of candidates to fetch from vector search before
                      graph expansion and reranking (default 100).
    """

    def __init__(self, top_k_hybrid: int = 100) -> None:
        settings = get_settings()

        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._collection = settings.qdrant_collection
        self._top_k_hybrid = top_k_hybrid

        embedder = Embedder(device=settings.bge_m3_device)
        self.hybrid = HybridRetriever(self._client, self._collection, embedder)
        self.expander = GraphExpander()
        # Reranker runs on CPU: BGE-M3 + reranker together exceed MPS VRAM budget.
        # Reranking only scores top-N pairs so CPU latency is acceptable.
        self.reranker = Reranker(device="cpu")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_n: int = 8) -> list[dict[str, Any]]:
        """Run the full three-stage retrieval pipeline.

        Steps:
        1. Hybrid dense+sparse retrieval → up to ``top_k_hybrid`` candidates.
        2. Graph expansion → fetch cross-refs and definition chunks.
        3. Rerank all candidates → return the best *top_n*.

        Each result dict has the following guaranteed keys::

            chunk_id      str   — e.g. "COMAR.15.05.01.06"
            citation      str   — e.g. "COMAR 15.05.01.06"
            chunk_text    str   — embeddable text (breadcrumb + regulation body)
            rerank_score  float — cross-encoder relevance score
            metadata      dict  — all other payload fields
            context_path  str   — human-readable breadcrumb

        Args:
            query: Natural-language question or keyword query.
            top_n: Number of results to return after reranking.

        Returns:
            List of result dicts, sorted by descending ``rerank_score``.
        """
        # ── Stage 1: Hybrid retrieval ─────────────────────────────────────
        candidates = self.hybrid.retrieve(query, top_k=self._top_k_hybrid)
        logger.info("Stage 1: %d hybrid candidates", len(candidates))

        # ── Stage 2: Graph expansion ──────────────────────────────────────
        candidate_ids = [c["chunk_id"] for c in candidates if c["chunk_id"]]
        expanded_ids = self.expander.expand(candidate_ids)

        if expanded_ids:
            expanded_chunks = self._fetch_by_chunk_ids(expanded_ids)
            existing_ids = {c["chunk_id"] for c in candidates}
            added = 0
            for chunk in expanded_chunks:
                if chunk["chunk_id"] not in existing_ids:
                    candidates.append(chunk)
                    existing_ids.add(chunk["chunk_id"])
                    added += 1
            logger.info("Stage 2: graph expansion added %d chunks", added)

        # ── Attach context paths ──────────────────────────────────────────
        for c in candidates:
            c["context_path"] = self.expander.get_context_path(c["chunk_id"])

        # ── Stage 3: Rerank ───────────────────────────────────────────────
        reranked = self.reranker.rerank(query, candidates, top_n=top_n)
        logger.info("Stage 3: reranked to %d results", len(reranked))

        # ── Normalise output schema ───────────────────────────────────────
        results: list[dict[str, Any]] = []
        for r in reranked:
            meta = r.get("metadata", {})
            results.append(
                {
                    "chunk_id": r.get("chunk_id", ""),
                    "citation": meta.get("citation", ""),
                    "chunk_text": r.get("chunk_text", ""),
                    "rerank_score": r.get("rerank_score", 0.0),
                    "metadata": meta,
                    "context_path": r.get("context_path", ""),
                }
            )
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch Qdrant payloads for a list of chunk_ids.

        Batches requests in groups of 50 to avoid overly large filter
        expressions.

        Args:
            chunk_ids: Regulation chunk_ids to retrieve.

        Returns:
            List of result dicts (same schema as :meth:`retrieve` candidates).
        """
        results: list[dict[str, Any]] = []
        batch_size = 50

        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i : i + batch_size]
            points, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="chunk_id",
                            match=qm.MatchAny(any=batch),
                        )
                    ]
                ),
                limit=len(batch),
                with_payload=True,
            )
            for point in points:
                payload = point.payload or {}
                results.append(
                    {
                        "chunk_id": payload.get("chunk_id", ""),
                        "chunk_text": payload.get("chunk_text", ""),
                        "rrf_score": 0.0,  # not from vector search
                        "metadata": {
                            k: v for k, v in payload.items() if k != "chunk_text"
                        },
                    }
                )

        return results
