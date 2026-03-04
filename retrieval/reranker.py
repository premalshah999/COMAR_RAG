"""retrieval/reranker.py — Cross-encoder reranking with BAAI/bge-reranker-v2-m3.

Takes a query and a list of candidate chunks (e.g. the top-100 from hybrid
retrieval + graph expansion) and re-scores every (query, chunk_text) pair
using a cross-encoder.  Cross-encoders are slower than bi-encoders but
significantly more accurate at judging relevance because they see both the
query and document together.

Model: ``BAAI/bge-reranker-v2-m3`` via :class:`sentence_transformers.CrossEncoder`.
Device is read from the ``BGE_M3_DEVICE`` environment variable (defaults to
``"cpu"``; use ``"mps"`` on Apple Silicon for a speedup).

Usage::

    from retrieval.reranker import Reranker

    reranker = Reranker()
    top_results = reranker.rerank(
        query="pesticide storage requirements",
        candidates=hybrid_results,   # list[dict] from HybridRetriever
        top_n=8,
    )
    # Each result dict gains a "rerank_score" key (float, higher = more relevant)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Cross-encoder reranker wrapping ``BAAI/bge-reranker-v2-m3``.

    The model is loaded lazily on the first call to :meth:`rerank`.

    Args:
        device: PyTorch device string — ``"cpu"``, ``"cuda"``, ``"mps"``.
                Defaults to the ``BGE_M3_DEVICE`` environment variable or
                ``"cpu"``.
    """

    def __init__(self, device: str | None = None) -> None:
        self.device: str = device or os.getenv("BGE_M3_DEVICE", "cpu")
        self._model: Any = None

    # ── Lazy loader ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading %s on device=%s …", _MODEL_NAME, self.device)
        self._model = CrossEncoder(
            _MODEL_NAME,
            device=self.device,
        )
        logger.info("Reranker model loaded.")

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 8,
        batch_size: int = 64,
    ) -> list[dict[str, Any]]:
        """Score and sort *candidates* by relevance to *query*.

        Each candidate dict must contain a ``"chunk_text"`` key.  The method
        adds a ``"rerank_score"`` key to every returned dict.

        Args:
            query: The user's natural-language question.
            candidates: List of chunk dicts (e.g. from
                        :class:`~retrieval.hybrid_retriever.HybridRetriever`).
            top_n: Number of top-scoring results to return.
            batch_size: Number of (query, text) pairs per forward pass.
                        Larger values improve GPU/CPU throughput; default 64.

        Returns:
            List of up to *top_n* candidate dicts, sorted by descending
            ``rerank_score``, each augmented with a ``"rerank_score"`` float.
        """
        if not candidates:
            return []

        self._load()

        pairs = [(query, c.get("chunk_text", "")) for c in candidates]
        scores = self._model.predict(pairs, batch_size=batch_size)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        results: list[dict[str, Any]] = []
        for candidate, score in ranked[:top_n]:
            result = dict(candidate)
            result["rerank_score"] = float(score)
            results.append(result)

        logger.debug(
            "Reranker: %d candidates → top %d selected (best score %.4f)",
            len(candidates),
            len(results),
            results[0]["rerank_score"] if results else 0.0,
        )
        return results
