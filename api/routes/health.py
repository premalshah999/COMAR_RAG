"""api/routes/health.py — /api/health and /api/stats endpoints."""
from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path

from fastapi import APIRouter

from api.config import get_settings
from api.models import HealthResponse, StatsResponse

router = APIRouter(prefix="/api", tags=["health"])


def _check_qdrant() -> tuple[bool, int, str]:
    """Synchronous Qdrant health check (runs in thread pool)."""
    settings = get_settings()
    try:
        from api.services.retriever import _get_qdrant
        c = _get_qdrant()
        info = c.get_collection(settings.qdrant_collection)
        points = info.points_count or 0
        status = "ok" if points > 0 else "degraded"
        return True, points, status
    except Exception:
        return False, 0, "degraded"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint — verifies Qdrant connection and LLM readiness."""
    settings = get_settings()
    
    # Run blocking Qdrant check in thread pool to avoid blocking event loop
    connected, points, status = await asyncio.to_thread(_check_qdrant)

    return HealthResponse(
        status=status,
        qdrant_connected=connected,
        qdrant_points=points,
        qdrant_collection=settings.qdrant_collection,
        llm_ready=settings.llm_ready,
        llm_model=settings.llm_model,
    )


def _load_stats() -> tuple[int, int, int]:
    """Synchronous stats loading (runs in thread pool)."""
    graph_nodes = 0
    graph_edges = 0
    defs = 0
    
    graph_path = Path("./data/comar_graph.pkl")
    if graph_path.exists():
        try:
            import networkx as nx
            with open(graph_path, "rb") as f:
                g: nx.DiGraph = pickle.load(f)
            graph_nodes = g.number_of_nodes()
            graph_edges = g.number_of_edges()
        except Exception:
            pass

    defs_path = Path("./data/definitions.json")
    if defs_path.exists():
        try:
            defs = len(json.loads(defs_path.read_text()))
        except Exception:
            pass
    
    return graph_nodes, graph_edges, defs


@router.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Corpus statistics endpoint — returns counts of regulations, chunks, graph nodes."""
    # Run blocking file I/O in thread pool
    graph_nodes, graph_edges, defs = await asyncio.to_thread(_load_stats)
    
    # Get actual chunk count from Qdrant if available
    chunks = 50827  # Default fallback
    try:
        from api.services.retriever import _get_qdrant
        from api.config import get_settings
        settings = get_settings()
        client = _get_qdrant()
        info = await asyncio.to_thread(client.get_collection, settings.qdrant_collection)
        chunks = info.points_count or chunks
    except Exception:
        pass

    return StatsResponse(
        regulations=3309,
        chunks=chunks,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        titles=["15", "26"],
        definitions=defs,
    )
