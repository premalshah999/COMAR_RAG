"""api/routes/health.py — /api/health and /api/stats endpoints."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from fastapi import APIRouter

from api.config import get_settings
from api.models import HealthResponse, StatsResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    points = 0
    connected = False
    try:
        from api.services.retriever import _get_qdrant
        c = _get_qdrant()
        info = c.get_collection(settings.qdrant_collection)
        points = info.points_count or 0
        connected = True
        status = "ok" if points > 0 else "degraded"
    except Exception:
        status = "degraded"

    return HealthResponse(
        status=status,
        qdrant_connected=connected,
        qdrant_points=points,
        qdrant_collection=settings.qdrant_collection,
        llm_ready=settings.llm_ready,
        llm_model=settings.llm_model,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    graph_nodes = 0
    graph_edges = 0
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

    defs = 0
    defs_path = Path("./data/definitions.json")
    if defs_path.exists():
        try:
            defs = len(json.loads(defs_path.read_text()))
        except Exception:
            pass

    return StatsResponse(
        regulations=3309,
        chunks=50827,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        titles=["15", "26"],
        definitions=defs,
    )
