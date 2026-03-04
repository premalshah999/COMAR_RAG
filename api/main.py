"""api/main.py — COMAR RAG FastAPI application.

Run with::

    uvicorn api.main:app --reload --port 8000

Environment variables for security/rate limiting:
    REQUIRE_API_KEY     — Set to "true" to require X-API-Key header (default: false)
    COMAR_API_KEYS      — Comma-separated list of valid API keys
    RATE_LIMIT_ENABLED  — Enable rate limiting (default: true)
    RATE_LIMIT_REQUESTS — Max requests per window (default: 60)
    RATE_LIMIT_WINDOW   — Window size in seconds (default: 60)
    CORS_ORIGINS        — Comma-separated allowed origins (default: localhost only)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api.routes.chat import router as chat_router
from api.routes.health import router as health_router
from api.middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
    RequestTracingMiddleware,
    configure_logging,
)

# Configure structured logging with request ID support
configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


# ── Startup/Shutdown Lifespan ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: pre-warm models on startup, cleanup on shutdown."""
    logger.info("🚀 Starting COMAR RAG API...")
    
    # Pre-warm the BGE-M3 embedder to avoid cold-start latency
    try:
        logger.info("Pre-warming BGE-M3 embedder...")
        from api.services.retriever import _get_embedder
        embedder = await asyncio.to_thread(_get_embedder)
        await asyncio.to_thread(embedder._load)
        logger.info("✓ BGE-M3 embedder ready")
    except Exception as exc:
        logger.warning("Embedder pre-warm failed (will load on first request): %s", exc)
    
    # Pre-warm the Qdrant client connection
    try:
        logger.info("Checking Qdrant connection...")
        from api.services.retriever import _get_qdrant
        from api.config import get_settings
        client = _get_qdrant()
        settings = get_settings()
        info = await asyncio.to_thread(client.get_collection, settings.qdrant_collection)
        logger.info("✓ Qdrant connected: %d vectors in '%s'", info.points_count or 0, settings.qdrant_collection)
    except Exception as exc:
        logger.warning("Qdrant connection check failed: %s", exc)
    
    # Pre-warm the knowledge graph
    try:
        logger.info("Loading knowledge graph...")
        from api.services.retriever import _get_expander
        expander = await asyncio.to_thread(_get_expander)
        if expander:
            logger.info("✓ Knowledge graph loaded: %d nodes", expander.graph.number_of_nodes())
    except Exception as exc:
        logger.warning("Knowledge graph load failed: %s", exc)
    
    logger.info("✅ COMAR RAG API ready to serve requests")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("👋 Shutting down COMAR RAG API...")


app = FastAPI(
    title="COMAR RAG API",
    description="Regulatory question-answering for Maryland COMAR Title 15 & 26",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# ── Security Middleware (order matters: outermost first) ─────────────────────
# Request tracing (adds X-Request-ID, logs timing)
app.add_middleware(RequestTracingMiddleware)

# Rate limiting (per-IP throttling)
app.add_middleware(RateLimitMiddleware)

# API key authentication (optional, controlled by REQUIRE_API_KEY)
app.add_middleware(APIKeyMiddleware)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Load allowed origins from environment, with sensible defaults for dev
_cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicit methods, not "*"
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(chat_router)

# ── Serve built React app in production ───────────────────────────────────────
_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="spa")
