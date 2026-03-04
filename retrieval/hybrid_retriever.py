"""retrieval/hybrid_retriever.py — Dense + sparse hybrid search with manual RRF fusion.

Performs two independent Qdrant searches (dense semantic + sparse BM25-like),
then fuses the ranked lists with Reciprocal Rank Fusion (k=60) to produce a
single relevance-ordered result list.

Usage::

    from qdrant_client import QdrantClient
    from ingestion.embedder import Embedder
    from retrieval.hybrid_retriever import HybridRetriever

    client  = QdrantClient(host="localhost", port=6333)
    embedder = Embedder()
    retriever = HybridRetriever(client, "comar_regulations", embedder)

    results = retriever.retrieve("pesticide storage requirements", top_k=100)
    # results: list of dicts with chunk_id, chunk_text, rrf_score, metadata
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ingestion.embedder import Embedder

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF constant


class HybridRetriever:
    """Dense + sparse hybrid search with Reciprocal Rank Fusion.

    Args:
        qdrant_client: Connected :class:`QdrantClient` instance.
        collection_name: Name of the Qdrant collection to search.
        embedder: Initialised :class:`~ingestion.embedder.Embedder` for
                  query vectorisation.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        embedder: Embedder,
    ) -> None:
        self.client = qdrant_client
        self.collection = collection_name
        self.embedder = embedder

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        """Hybrid-search *collection* for *query* and return RRF-fused results.

        Steps:
        1. Embed query → dense vector + sparse weight dict.
        2. Dense ANN search  (``"dense"`` named vector, cosine).
        3. Sparse BM25 search (``"sparse"`` named vector).
        4. RRF fusion with k=60 across both ranked lists.
        5. Return top-k dicts sorted by descending rrf_score.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return.

        Returns:
            List of result dicts::

                {
                    "chunk_id":   str,
                    "chunk_text": str,
                    "rrf_score":  float,
                    "metadata":   dict,   # all payload fields except chunk_text
                }
        """
        # ── 1. Embed ──────────────────────────────────────────────────────
        dense_vec = self.embedder.embed_dense([query])[0]
        sparse_vec = self.embedder.embed_sparse([query])[0]

        # ── 2. Dense search ───────────────────────────────────────────────
        dense_hits = self.client.query_points(
            collection_name=self.collection,
            query=dense_vec,
            using="dense",
            limit=top_k,
            with_payload=True,
        ).points

        # ── 3. Sparse search ──────────────────────────────────────────────
        sparse_hits = self.client.query_points(
            collection_name=self.collection,
            query=qm.SparseVector(
                indices=list(sparse_vec.keys()),
                values=list(sparse_vec.values()),
            ),
            using="sparse",
            limit=top_k,
            with_payload=True,
        ).points

        # ── 4. RRF fusion ─────────────────────────────────────────────────
        rrf_scores: dict[str, float] = {}
        point_data: dict[str, Any] = {}

        for rank, point in enumerate(dense_hits):
            pid = str(point.id)
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            point_data[pid] = point

        for rank, point in enumerate(sparse_hits):
            pid = str(point.id)
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if pid not in point_data:
                point_data[pid] = point

        # ── 5. Sort and build output ──────────────────────────────────────
        sorted_pids = sorted(
            rrf_scores, key=lambda x: rrf_scores[x], reverse=True
        )[:top_k]

        results: list[dict[str, Any]] = []
        for pid in sorted_pids:
            point = point_data[pid]
            payload = point.payload or {}
            results.append(
                {
                    "chunk_id": payload.get("chunk_id", ""),
                    "chunk_text": payload.get("chunk_text", ""),
                    "rrf_score": round(rrf_scores[pid], 6),
                    "metadata": {
                        k: v for k, v in payload.items() if k != "chunk_text"
                    },
                }
            )

        logger.debug(
            "HybridRetriever: %d dense + %d sparse → %d fused results",
            len(dense_hits),
            len(sparse_hits),
            len(results),
        )
        return results

    def search_by_citation(self, citation: str) -> dict[str, Any] | None:
        """Fetch a regulation by its exact COMAR citation string.

        Uses a payload filter (exact match on the ``citation`` keyword field)
        rather than vector search, so no embedding is required.

        Args:
            citation: Canonical citation, e.g. ``"COMAR 15.05.01.06"``.

        Returns:
            A single result dict (same schema as :meth:`retrieve`) or ``None``
            if no matching regulation is found.
        """
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="citation",
                        match=qm.MatchValue(value=citation),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )

        if not points:
            logger.debug("search_by_citation: no match for '%s'", citation)
            return None

        payload = points[0].payload or {}
        return {
            "chunk_id": payload.get("chunk_id", ""),
            "chunk_text": payload.get("chunk_text", ""),
            "rrf_score": 1.0,
            "metadata": {k: v for k, v in payload.items() if k != "chunk_text"},
        }
