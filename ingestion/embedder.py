"""embedder.py — BGE-M3 embedding wrapper for COMAR RAG.

Wraps BAAI/bge-m3 via :class:`FlagEmbedding.BGEM3FlagModel` to produce
three complementary vector representations per text:

Dense   (1024-dim float)    — standard semantic similarity
Sparse  ({token_id: weight}) — BM25-equivalent lexical matching
ColBERT (list of 1024-dim)   — late-interaction token-level matching

The model is loaded lazily on first use and cached for the lifetime of the
:class:`Embedder` instance.  Batch size and device are configurable via
environment variables (``BGE_M3_DEVICE``, default ``"cpu"``).

Usage::

    from ingestion.embedder import Embedder

    emb = Embedder()
    dense, sparse, colbert = emb.embed_all(["text one", "text two"])
    # dense:   list[list[float]]          length = n_texts, each 1024-dim
    # sparse:  list[dict[int, float]]     BM25 token weights
    # colbert: list[list[list[float]]]    token-level vectors
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 32
_MODEL_NAME = "BAAI/bge-m3"


class Embedder:
    """Thin wrapper around :class:`FlagEmbedding.BGEM3FlagModel`.

    The underlying model is loaded on the first call to any ``embed_*`` method
    (lazy initialisation) so that imports are cheap.

    Args:
        device: PyTorch device string — ``"cpu"``, ``"cuda"``, ``"mps"``.
                Defaults to the ``BGE_M3_DEVICE`` environment variable or
                ``"cpu"``.
        batch_size: Number of texts per inference batch.
        use_fp16: Use FP16 precision where supported (speeds up GPU inference).
    """

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        use_fp16: bool = False,
    ) -> None:
        self.device: str = device or os.getenv("BGE_M3_DEVICE", "cpu")
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self._model: Any = None  # loaded lazily

    # ── Lazy model loader ─────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the BGE-M3 model if not already loaded."""
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required. Install it with: pip install FlagEmbedding"
            ) from exc

        logger.info(
            "Loading %s on device=%s, fp16=%s …", _MODEL_NAME, self.device, self.use_fp16
        )
        self._model = BGEM3FlagModel(
            _MODEL_NAME,
            use_fp16=self.use_fp16,
            device=self.device,
        )
        logger.info("BGE-M3 model loaded.")

    # ── Batched inference helpers ─────────────────────────────────────────────

    def _iter_batches(
        self, texts: list[str], desc: str = "Embedding"
    ) -> list[dict[str, Any]]:
        """Run encode() on *texts* in batches, return raw model outputs.

        Args:
            texts: Input strings to embed.
            desc: tqdm progress bar label.

        Returns:
            List of per-batch output dicts from ``BGEM3FlagModel.encode()``.
        """
        self._load()
        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        results: list[dict[str, Any]] = []
        for batch in tqdm(batches, desc=desc, unit="batch", leave=False):
            output = self._model.encode(
                batch,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
            )
            results.append(output)
        return results

    @staticmethod
    def _concat_dense(
        batch_outputs: list[dict[str, Any]],
    ) -> list[list[float]]:
        """Flatten batched dense vectors into a single list."""
        dense: list[list[float]] = []
        for batch in batch_outputs:
            vecs = batch.get("dense_vecs", [])
            # BGEM3FlagModel returns numpy arrays; convert to plain Python lists
            for vec in vecs:
                dense.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))
        return dense

    @staticmethod
    def _concat_sparse(
        batch_outputs: list[dict[str, Any]],
    ) -> list[dict[int, float]]:
        """Flatten batched sparse vectors into a single list.

        BGEM3FlagModel returns sparse vectors as list-of-dicts where keys are
        token ids (ints) and values are weights (floats).
        """
        sparse: list[dict[int, float]] = []
        for batch in batch_outputs:
            for sv in batch.get("lexical_weights", []):
                # sv may be a defaultdict; cast to plain dict with int keys
                sparse.append({int(k): float(v) for k, v in sv.items()})
        return sparse

    @staticmethod
    def _concat_colbert(
        batch_outputs: list[dict[str, Any]],
    ) -> list[list[list[float]]]:
        """Flatten batched ColBERT vectors into a single list."""
        colbert: list[list[list[float]]] = []
        for batch in batch_outputs:
            for token_vecs in batch.get("colbert_vecs", []):
                # token_vecs: (n_tokens, 1024) numpy array
                colbert.append(
                    [v.tolist() if hasattr(v, "tolist") else list(v) for v in token_vecs]
                )
        return colbert

    # ── Public embed methods ──────────────────────────────────────────────────

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* and return only the dense (1024-dim) vectors.

        Args:
            texts: Input strings.

        Returns:
            List of 1024-dimensional float vectors.
        """
        batches = self._iter_batches(texts, desc="Dense embed")
        return self._concat_dense(batches)

    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        """Embed *texts* and return only the sparse (BM25-like) weight dicts.

        Args:
            texts: Input strings.

        Returns:
            List of ``{token_id: weight}`` dicts, one per input text.
        """
        batches = self._iter_batches(texts, desc="Sparse embed")
        return self._concat_sparse(batches)

    def embed_colbert(self, texts: list[str]) -> list[list[list[float]]]:
        """Embed *texts* and return token-level ColBERT vectors.

        Args:
            texts: Input strings.

        Returns:
            List of ``(n_tokens × 1024)`` float matrices.
        """
        batches = self._iter_batches(texts, desc="ColBERT embed")
        return self._concat_colbert(batches)

    def embed_all(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[int, float]], list[list[list[float]]]]:
        """Embed *texts* and return all three vector types in one pass.

        This is the most efficient method because the model runs only once per
        batch, producing all three representations simultaneously.

        Args:
            texts: Input strings.

        Returns:
            3-tuple ``(dense, sparse, colbert)`` where:

            - ``dense``   : ``list[list[float]]``             length=n, each 1024-dim
            - ``sparse``  : ``list[dict[int, float]]``        BM25 token weights
            - ``colbert`` : ``list[list[list[float]]]``       token-level vectors
        """
        batches = self._iter_batches(texts, desc="BGE-M3 embed")
        return (
            self._concat_dense(batches),
            self._concat_sparse(batches),
            self._concat_colbert(batches),
        )
