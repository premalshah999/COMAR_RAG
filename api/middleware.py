"""api/middleware.py — Security and observability middleware for COMAR RAG.

Provides:
- API key authentication (optional, controlled by REQUIRE_API_KEY env var)
- Rate limiting per client IP
- Request ID tracing for all requests
- Structured logging with request context
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from collections import defaultdict
from contextvars import ContextVar
from functools import wraps
from typing import Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ── Context Variables for Request Tracing ─────────────────────────────────────

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Get the current request ID from context."""
    return request_id_var.get()


# ── API Key Authentication ────────────────────────────────────────────────────

# Load valid API keys from environment (comma-separated)
_API_KEYS: set[str] = set()
_raw_keys = os.getenv("COMAR_API_KEYS", "")
if _raw_keys:
    _API_KEYS = {k.strip() for k in _raw_keys.split(",") if k.strip()}

# Whether to require API key authentication
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() in ("true", "1", "yes")

# Paths that don't require authentication
_PUBLIC_PATHS: set[str] = {
    "/api/health",
    "/api/docs",
    "/api/redoc",
    "/api/openapi.json",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate API key in X-API-Key header or api_key query param.
    
    Skips validation when:
    - REQUIRE_API_KEY is false (default for development)
    - The request path is in _PUBLIC_PATHS
    - No API keys are configured
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth if not required or no keys configured
        if not REQUIRE_API_KEY or not _API_KEYS:
            return await call_next(request)

        # Skip auth for public paths
        path = request.url.path.rstrip("/") or "/"
        if path in _PUBLIC_PATHS or path.startswith("/api/docs") or path.startswith("/api/redoc"):
            return await call_next(request)

        # Check header first, then query param
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")

        if not api_key:
            logger.warning("Request without API key: %s %s", request.method, path)
            return JSONResponse(
                status_code=401,
                content={"detail": "API key required. Provide X-API-Key header or api_key query param."},
            )

        if api_key not in _API_KEYS:
            logger.warning("Invalid API key attempt: %s %s", request.method, path)
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key."},
            )

        return await call_next(request)


# ── Rate Limiting ─────────────────────────────────────────────────────────────

# Rate limit configuration from environment
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))  # requests per window
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # window in seconds
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")

# In-memory rate limit store: {client_ip: [(timestamp, count), ...]}
_rate_limit_store: dict[str, list[tuple[float, int]]] = defaultdict(list)


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For for proxied requests."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _clean_old_entries(entries: list[tuple[float, int]], window: float) -> list[tuple[float, int]]:
    """Remove entries older than the rate limit window."""
    cutoff = time.time() - window
    return [(ts, count) for ts, count in entries if ts > cutoff]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple sliding window rate limiter.
    
    Limits requests per IP address within a configurable time window.
    Returns 429 Too Many Requests when limit is exceeded.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not RATE_LIMIT_ENABLED:
            return await call_next(request)

        # Skip rate limiting for health checks
        if request.url.path in ("/api/health", "/health"):
            return await call_next(request)

        client_ip = _get_client_ip(request)
        current_time = time.time()

        # Clean old entries and count requests in window
        entries = _clean_old_entries(_rate_limit_store[client_ip], RATE_LIMIT_WINDOW)
        request_count = sum(count for _, count in entries)

        if request_count >= RATE_LIMIT_REQUESTS:
            # Find when the oldest entry expires
            if entries:
                retry_after = int(entries[0][0] + RATE_LIMIT_WINDOW - current_time) + 1
            else:
                retry_after = RATE_LIMIT_WINDOW

            logger.warning(
                "Rate limit exceeded for %s: %d requests in %ds window",
                client_ip, request_count, RATE_LIMIT_WINDOW
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        # Record this request
        entries.append((current_time, 1))
        _rate_limit_store[client_ip] = entries

        response = await call_next(request)

        # Add rate limit headers
        remaining = max(0, RATE_LIMIT_REQUESTS - request_count - 1)
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(RATE_LIMIT_WINDOW)

        return response


# ── Request Tracing ───────────────────────────────────────────────────────────

class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Add request ID to all requests and log request/response details."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request_id_var.set(request_id)

        start_time = time.perf_counter()
        client_ip = _get_client_ip(request)

        # Log request
        logger.info(
            "[%s] %s %s from %s",
            request_id, request.method, request.url.path, client_ip
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(
                "[%s] %s %s → 500 in %.0fms: %s",
                request_id, request.method, request.url.path, elapsed, exc
            )
            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        # Log response
        logger.info(
            "[%s] %s %s → %d in %.0fms",
            request_id, request.method, request.url.path, response.status_code, elapsed
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


# ── Structured Logging Filter ─────────────────────────────────────────────────

class RequestIDFilter(logging.Filter):
    """Add request_id to all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure structured logging with request ID context."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  [%(request_id)s]  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Add filter to root logger
    logging.getLogger().addFilter(RequestIDFilter())
