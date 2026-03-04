"""api/models.py — Pydantic request/response models."""
from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    filters: dict[str, list[str]] = Field(
        default_factory=dict,
        description="e.g. {'title_num': ['15']} to restrict search",
    )
    top_k: int = Field(default=10, ge=1, le=20)


class Source(BaseModel):
    citation: str
    title_name: str
    subtitle_name: str
    chapter_name: str
    regulation_name: str
    text_snippet: str
    score: float
    chunk_type: str
    effective_date: str = ""
    context_path: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    conversation_id: str
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    mode: Literal["live", "stub"] = "stub"


# SSE token event (streamed line-by-line)
class TokenEvent(BaseModel):
    token: str = ""
    done: bool = False
    sources: list[Source] = []
    conversation_id: str = ""
    message_id: str = ""
    mode: Literal["live", "stub"] = "stub"


# ── Health / Stats ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    qdrant_connected: bool
    qdrant_points: int
    qdrant_collection: str
    llm_ready: bool
    llm_model: str


class StatsResponse(BaseModel):
    regulations: int
    chunks: int
    graph_nodes: int
    graph_edges: int
    titles: list[str]
    definitions: int


# ── Search ─────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1024)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: dict[str, list[str]] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[Source]
    query: str
    retrieval_ms: float
