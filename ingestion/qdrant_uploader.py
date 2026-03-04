"""qdrant_uploader.py — Create the Qdrant collection and upload COMAR chunks.

Collection schema
-----------------
Named vectors:

* ``"dense"``  — 1024-dim Cosine, from BGE-M3 dense encoder
* ``"sparse"`` — SparseIndexParams, from BGE-M3 lexical (BM25-equivalent) encoder

Payload indexes for fast filtered search:

* ``title_num``      (keyword)
* ``chunk_type``     (keyword)
* ``effective_date`` (keyword)
* ``chunk_id``       (keyword)

Point IDs are deterministic ``uuid5`` values derived from each chunk's
``chunk_id`` string, ensuring idempotent re-ingestion.

Usage::

    from ingestion.embedder import Embedder
    from ingestion.qdrant_uploader import upload_chunks, verify_collection

    emb = Embedder()
    upload_chunks(chunks, emb)
    stats = verify_collection()
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

from ingestion.embedder import Embedder

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_UPLOAD_BATCH_SIZE = 100
_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID namespace URL

DENSE_DIM = 1024
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def _get_client() -> QdrantClient:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def _get_collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION", "comar_regulations")


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Derive a deterministic UUID from a chunk_id string."""
    return str(uuid.uuid5(_UUID_NAMESPACE, chunk_id))


# ── Collection management ─────────────────────────────────────────────────────


def ensure_collection(client: QdrantClient | None = None) -> None:
    """Create the Qdrant collection if it doesn't already exist.

    Configures:
    - Named dense vector (1024-dim, Cosine distance)
    - Named sparse vector (IDF modifier for BM25-like scoring)
    - Payload indexes on ``title_num``, ``chunk_type``, ``effective_date``,
      ``chunk_id``

    Args:
        client: Optional pre-built :class:`QdrantClient`.  A new one is
                created from environment variables if ``None``.
    """
    client = client or _get_client()
    name = _get_collection_name()

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        logger.info("Collection '%s' already exists — skipping creation.", name)
        return

    logger.info("Creating collection '%s' …", name)
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR_NAME: qm.VectorParams(
                size=DENSE_DIM,
                distance=qm.Distance.COSINE,
                on_disk=False,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                index=qm.SparseIndexParams(
                    on_disk=False,
                ),
            ),
        },
        optimizers_config=qm.OptimizersConfigDiff(
            indexing_threshold=20_000,
        ),
    )

    # ── Payload indexes ────────────────────────────────────────────────────
    indexed_fields = [
        ("title_num", qm.PayloadSchemaType.KEYWORD),
        ("chunk_type", qm.PayloadSchemaType.KEYWORD),
        ("effective_date", qm.PayloadSchemaType.KEYWORD),
        ("chunk_id", qm.PayloadSchemaType.KEYWORD),
        ("citation", qm.PayloadSchemaType.KEYWORD),
    ]
    for field, schema in indexed_fields:
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema,
        )
        logger.debug("Payload index created: %s (%s)", field, schema)

    logger.info("Collection '%s' created successfully.", name)


# ── Upload ─────────────────────────────────────────────────────────────────────


def _build_points(
    batch: list[dict[str, Any]],
    dense_vecs: list[list[float]],
    sparse_vecs: list[dict[int, float]],
) -> list[qm.PointStruct]:
    """Construct :class:`qdrant_client.http.models.PointStruct` objects.

    Args:
        batch: Chunk dicts for this batch.
        dense_vecs: Parallel list of 1024-dim dense vectors.
        sparse_vecs: Parallel list of ``{token_id: weight}`` dicts.

    Returns:
        List of :class:`~qdrant_client.http.models.PointStruct` ready for upsert.
    """
    points: list[qm.PointStruct] = []
    for chunk, dvec, svec in zip(batch, dense_vecs, sparse_vecs):
        pid = _chunk_id_to_uuid(chunk["chunk_id"])

        # Build payload — exclude chunk_text (large) but keep everything else
        payload = {
            k: v
            for k, v in chunk.items()
            if k not in ("chunk_text",)
        }
        # Stash the chunk text separately so it can be returned in results
        payload["chunk_text"] = chunk.get("chunk_text", "")

        # Convert sparse dict to Qdrant SparseVector
        sparse_indices = list(svec.keys())
        sparse_values = [svec[i] for i in sparse_indices]

        points.append(
            qm.PointStruct(
                id=pid,
                vector={
                    DENSE_VECTOR_NAME: dvec,
                    SPARSE_VECTOR_NAME: qm.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                },
                payload=payload,
            )
        )
    return points


def upload_chunks(
    chunks: list[dict[str, Any]],
    embedder: Embedder,
    client: QdrantClient | None = None,
    batch_size: int = _UPLOAD_BATCH_SIZE,
) -> None:
    """Embed and upsert all chunks into Qdrant.

    Processes chunks in batches of *batch_size*:

    1. Extract ``chunk_text`` for the current batch.
    2. Call :meth:`~ingestion.embedder.Embedder.embed_all` (dense + sparse in
       one forward pass — ColBERT vectors are not stored in Qdrant for now
       due to storage cost, but can be added later).
    3. Build :class:`qdrant_client.http.models.PointStruct` objects.
    4. Upsert to Qdrant.

    Args:
        chunks: Chunk dicts from :func:`~ingestion.chunker.create_chunks`.
        embedder: Initialised :class:`~ingestion.embedder.Embedder`.
        client: Optional pre-built :class:`QdrantClient`.
        batch_size: Number of chunks per upsert batch.
    """
    client = client or _get_client()
    collection = _get_collection_name()

    ensure_collection(client)

    total = len(chunks)
    uploaded = 0
    batches = [chunks[i : i + batch_size] for i in range(0, total, batch_size)]

    for batch in tqdm(batches, desc="Uploading to Qdrant", unit="batch"):
        texts = [c["chunk_text"] for c in batch]

        # Run embedding (dense + sparse only; skip colbert to save memory)
        embedder._load()
        raw = embedder._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = [
            v.tolist() if hasattr(v, "tolist") else list(v)
            for v in raw["dense_vecs"]
        ]
        sparse_vecs = [
            {int(k): float(v) for k, v in sv.items()}
            for sv in raw["lexical_weights"]
        ]

        points = _build_points(batch, dense_vecs, sparse_vecs)

        client.upsert(
            collection_name=collection,
            points=points,
            wait=True,
        )
        uploaded += len(points)

    logger.info(
        "Upload complete: %d / %d chunks upserted to '%s'",
        uploaded,
        total,
        collection,
    )


# ── Verification ───────────────────────────────────────────────────────────────


def verify_collection(client: QdrantClient | None = None) -> dict[str, Any]:
    """Return a summary dict with collection statistics.

    Args:
        client: Optional pre-built :class:`QdrantClient`.

    Returns:
        Dict with keys: ``name``, ``vectors_count``, ``points_count``,
        ``indexed_fields``, ``status``.
    """
    client = client or _get_client()
    name = _get_collection_name()
    info = client.get_collection(name)

    result = {
        "name": name,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "indexed_fields": list(info.payload_schema.keys())
        if info.payload_schema
        else [],
        "status": str(info.status),
    }
    logger.info(
        "Collection '%s': %s points, status=%s",
        name,
        result["points_count"],
        result["status"],
    )
    return result
