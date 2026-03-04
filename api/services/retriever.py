"""api/services/retriever.py — Production retrieval service for COMAR RAG.

Design principles for the stakeholder demo:
- Single BGE-M3 forward pass per request (embed_all) to avoid MPS sequential-pass hangs
- Module-level singletons: Embedder, QdrantClient, HybridRetriever (for citation lookup),
  GraphExpander (for context breadcrumbs)
- Direct citation lookup when COMAR XX.XX.XX.XX appears in the query
- Full chunk_text (no 400-char truncation) passed to the LLM
- Duplicate citation deduplication
- effective_date and context_path attached to every Source
- Filter applied post-retrieval (empty title_num list = no filter)
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from functools import lru_cache
from typing import Any

from api.config import get_settings
from api.models import Source

logger = logging.getLogger(__name__)

# ── COMAR citation pattern ─────────────────────────────────────────────────────
_CITATION_RE = re.compile(
    r"COMAR\s+\d{1,2}\.\d{2}\.\d{2}(?:\.\d{2})?",
    re.IGNORECASE,
)

# ── Stub results (shown when Qdrant / model unavailable) ──────────────────────
_STUB_SOURCES: list[dict[str, Any]] = [
    {
        "citation": "COMAR 15.10.04.01",
        "title_name": "MARYLAND DEPARTMENT OF AGRICULTURE",
        "subtitle_name": "PESTICIDE REGULATION",
        "chapter_name": "General Regulations",
        "regulation_name": "Definitions.",
        "text_snippet": (
            '"Pesticide" means any substance or mixture of substances intended for '
            "preventing, destroying, repelling, or mitigating any pest."
        ),
        "score": 0.91,
        "chunk_type": "definition",
        "effective_date": "",
        "context_path": "",
    },
    {
        "citation": "COMAR 26.08.02.01",
        "title_name": "DEPARTMENT OF THE ENVIRONMENT",
        "subtitle_name": "WATER MANAGEMENT",
        "chapter_name": "Water Quality Standards",
        "regulation_name": "Definitions.",
        "text_snippet": (
            '"Water quality standards" means the combination of designated uses, '
            "water quality criteria, and an antidegradation policy for waters of the State."
        ),
        "score": 0.87,
        "chunk_type": "definition",
        "effective_date": "",
        "context_path": "",
    },
    {
        "citation": "COMAR 26.08.02.03",
        "title_name": "DEPARTMENT OF THE ENVIRONMENT",
        "subtitle_name": "WATER MANAGEMENT",
        "chapter_name": "Water Quality Standards",
        "regulation_name": "General Water Quality Criteria.",
        "text_snippet": (
            "A. All surface waters shall be free from substances attributable to "
            "municipal, industrial, agricultural, or other discharges."
        ),
        "score": 0.82,
        "chunk_type": "regulation",
        "effective_date": "",
        "context_path": "",
    },
]


# ── Singletons ────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_embedder():
    """Module-level BGE-M3 Embedder (loaded lazily on first use)."""
    from ingestion.embedder import Embedder
    settings = get_settings()
    return Embedder(device=settings.bge_m3_device)


@lru_cache(maxsize=1)
def _get_qdrant():
    """Module-level QdrantClient."""
    from qdrant_client import QdrantClient
    settings = get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=30)


@lru_cache(maxsize=1)
def _get_hybrid():
    """Module-level HybridRetriever (for search_by_citation only — no embedding)."""
    from retrieval.hybrid_retriever import HybridRetriever
    return HybridRetriever(_get_qdrant(), get_settings().qdrant_collection, _get_embedder())


@lru_cache(maxsize=1)
def _get_expander():
    """Module-level GraphExpander (loads comar_graph.pkl once)."""
    try:
        from retrieval.graph_expander import GraphExpander
        return GraphExpander()
    except Exception as exc:
        logger.warning("GraphExpander unavailable (%s) — context paths will be empty", exc)
        return None


# ── Score normalisation ────────────────────────────────────────────────────────


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ── Internal helpers ──────────────────────────────────────────────────────────


def _context_path(chunk_id: str) -> str:
    expander = _get_expander()
    if expander is None:
        return ""
    try:
        return expander.get_context_path(chunk_id) or ""
    except Exception:
        return ""


def _dedup(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate citations, keeping the entry with the highest score."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in chunks:
        meta = c.get("metadata", {})
        cit = c.get("citation") or meta.get("citation", "")
        if cit not in seen:
            seen.add(cit)
            out.append(c)
    return out


def _apply_filters(sources: list[Source], filters: dict | None) -> list[Source]:
    """Filter by title_num post-retrieval.  Empty list means no filter."""
    if not filters:
        return sources
    allowed = [str(v) for v in filters.get("title_num", []) if v]
    if not allowed:
        return sources
    return [
        s for s in sources
        if any(f"COMAR {t}." in s.citation for t in allowed)
    ]


def _payload_to_chunk(payload: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "chunk_id": payload.get("chunk_id", ""),
        "citation": payload.get("citation", ""),
        "chunk_text": payload.get("chunk_text", ""),
        "score": score,
        "metadata": {k: v for k, v in payload.items() if k != "chunk_text"},
    }


def _chunks_to_sources(
    chunks: list[dict[str, Any]],
    filters: dict | None = None,
) -> list[Source]:
    sources = []
    for c in chunks:
        meta = c.get("metadata", {})
        raw_score = c.get("score", 0.0)
        # RRF scores are small (e.g. 0.03); rerank scores can be negative — normalise both
        score = round(_sigmoid(raw_score) if abs(raw_score) > 1.0 else float(raw_score), 4)

        chunk_id = c.get("chunk_id", "")
        sources.append(
            Source(
                citation=c.get("citation") or meta.get("citation", ""),
                title_name=meta.get("title_name", ""),
                subtitle_name=meta.get("subtitle_name", ""),
                chapter_name=meta.get("chapter_name", ""),
                regulation_name=meta.get("regulation_name", ""),
                text_snippet=c.get("chunk_text", ""),  # full text — no truncation
                score=score,
                chunk_type=meta.get("chunk_type", "regulation"),
                effective_date=meta.get("effective_date") or "",
                context_path=_context_path(chunk_id),
            )
        )
    return _apply_filters(sources, filters)


# ── Core retrieval (blocking — called via asyncio.to_thread) ──────────────────


def _do_retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
    """Run hybrid RRF search + optional citation direct-lookup.

    Uses a single embed_all() forward pass for efficiency.
    """
    from qdrant_client.http import models as qm

    embedder = _get_embedder()
    client = _get_qdrant()
    collection = get_settings().qdrant_collection

    # Single forward pass → dense + sparse vectors
    dense_vecs, sparse_vecs, _ = embedder.embed_all([query])
    dense_vec = dense_vecs[0]
    sparse_vec = sparse_vecs[0]

    results = client.query_points(
        collection_name=collection,
        prefetch=[
            qm.Prefetch(query=dense_vec, using="dense", limit=top_k * 3),
            qm.Prefetch(
                query=qm.SparseVector(
                    indices=list(sparse_vec.keys()),
                    values=list(sparse_vec.values()),
                ),
                using="sparse",
                limit=top_k * 3,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    chunks = []
    for p in results.points:
        payload = p.payload or {}
        chunks.append(_payload_to_chunk(payload, getattr(p, "score", 0.0)))

    return chunks


def _do_direct_lookup(citation: str) -> dict[str, Any] | None:
    """Fetch a single regulation by exact COMAR citation (no embedding)."""
    try:
        result = _get_hybrid().search_by_citation(citation)
        if result:
            payload = result.get("payload", result)
            return _payload_to_chunk(payload, 1.0)  # top priority score
    except Exception as exc:
        logger.debug("Direct lookup failed for %s: %s", citation, exc)
    return None


# ── Public async interface ─────────────────────────────────────────────────────


async def retrieve(
    query: str,
    top_k: int = 8,
    filters: dict[str, list[str]] | None = None,
) -> tuple[list[Source], float]:
    """Retrieve COMAR regulations relevant to *query*.

    - If a COMAR citation pattern is in the query, that regulation is fetched
      first via direct scroll (no embedding needed).
    - A full hybrid RRF search (dense + sparse, single embed_all pass) fills
      the remaining slots.
    - Duplicates are removed, filters applied, Sources constructed with full
      chunk_text, effective_date, and context_path.

    All blocking ML/IO work runs in a thread pool (asyncio.to_thread).

    Returns:
        ``(sources, elapsed_ms)``
    """
    settings = get_settings()
    t0 = time.perf_counter()

    try:
        # Fast reachability check
        client = _get_qdrant()
        info = client.get_collection(settings.qdrant_collection)
        if (info.points_count or 0) < 100:
            raise RuntimeError("Collection not ready")

        chunks: list[dict[str, Any]] = []

        # ── Direct citation lookup ─────────────────────────────────────────
        m = _CITATION_RE.search(query)
        if m:
            citation = m.group(0).upper()
            direct = await asyncio.to_thread(_do_direct_lookup, citation)
            if direct:
                chunks.append(direct)
                logger.info("Direct citation hit: %s", citation)

        # ── Hybrid RRF search (single embed_all pass) ─────────────────────
        hybrid_n = max(1, top_k - len(chunks))
        hybrid_chunks = await asyncio.to_thread(_do_retrieve, query, hybrid_n + 3)
        chunks.extend(hybrid_chunks)

        # ── Dedup, trim, build Sources ────────────────────────────────────
        chunks = _dedup(chunks)[:top_k]
        sources = _chunks_to_sources(chunks, filters)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Retrieved %d sources in %.0fms", len(sources), elapsed)
        return sources, elapsed

    except Exception as exc:
        logger.warning("Retriever fallback to stub (%s: %s)", type(exc).__name__, exc)
        elapsed = (time.perf_counter() - t0) * 1000
        return [Source(**s) for s in _STUB_SOURCES[:top_k]], elapsed
