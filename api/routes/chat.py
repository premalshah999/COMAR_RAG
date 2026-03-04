"""api/routes/chat.py — Streaming chat endpoint via Server-Sent Events."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.models import ChatRequest, SearchRequest, SearchResponse, Source
from api.services.intent import classify as classify_intent
from api.services.llm import generate_stream
from api.services.retriever import retrieve

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)

try:
    from pipeline.citation_verifier import CitationVerifier as _CitationVerifier
    _verifier = _CitationVerifier()
except Exception:
    _verifier = None

# ── Bounded conversation store with LRU eviction + TTL ──────────────────────
_MAX_CONVERSATIONS = int(os.getenv("MAX_CONVERSATIONS", "500"))
_CONVERSATION_TTL = int(os.getenv("CONVERSATION_TTL", "3600"))  # 1 hour default
_conversations: dict[str, dict] = {}  # {conv_id: {"messages": [...], "ts": float}}
_conv_lock = threading.Lock()
_last_cleanup = time.time()
_CLEANUP_INTERVAL = 300  # Clean up expired entries every 5 minutes


def _cleanup_expired() -> int:
    """Remove conversations older than TTL. Returns count of removed entries."""
    global _last_cleanup
    now = time.time()
    
    # Only run cleanup periodically
    if now - _last_cleanup < _CLEANUP_INTERVAL:
        return 0
    
    _last_cleanup = now
    cutoff = now - _CONVERSATION_TTL
    expired = [cid for cid, entry in _conversations.items() if entry["ts"] < cutoff]
    
    for cid in expired:
        del _conversations[cid]
    
    if expired:
        logger.debug("Cleaned up %d expired conversations", len(expired))
    
    return len(expired)


def _get_history(conv_id: str) -> list[dict]:
    with _conv_lock:
        # Periodic cleanup of expired entries
        _cleanup_expired()
        
        entry = _conversations.get(conv_id)
        if entry:
            entry["ts"] = time.time()
            return entry["messages"]
        
        # LRU eviction if at capacity
        if len(_conversations) >= _MAX_CONVERSATIONS:
            oldest = min(_conversations, key=lambda k: _conversations[k]["ts"])
            del _conversations[oldest]
            logger.debug("Evicted oldest conversation %s (LRU)", oldest[:8])
        
        _conversations[conv_id] = {"messages": [], "ts": time.time()}
        return _conversations[conv_id]["messages"]


def _append_history(conv_id: str, user_msg: str, assistant_msg: str) -> None:
    with _conv_lock:
        entry = _conversations.get(conv_id)
        if not entry:
            return
        entry["messages"].extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ])
        # Keep last 6 turns (12 messages) to stay within DeepSeek context limits
        entry["messages"] = entry["messages"][-12:]
        entry["ts"] = time.time()


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Events data line."""
    return f"data: {json.dumps(data)}\n\n"


# Pronouns/demonstratives that make a follow-up query ambiguous without context
_VAGUE_WORDS = frozenset({
    "they", "them", "their", "it", "its",
    "this", "that", "these", "those",
    "he", "she", "we", "such",
})

_REGULATORY_WORDS = frozenset({
    "comar", "regulation", "title", "section", "subtitle", "chapter",
    "pesticide", "applicator", "permit", "license", "certificate",
    "effluent", "discharge", "water", "air", "hazardous", "waste",
})


def _retrieval_query(message: str, history: list[dict]) -> str:
    """Return an enriched retrieval query for vague follow-up messages.

    When a message is short or uses pronouns ("they", "it", "this"), the
    previous user turn is prepended so Qdrant retrieves contextually
    relevant regulations rather than searching a meaningless fragment.

    The LLM always receives the *original* message — this only affects
    what is sent to the vector database.
    """
    if not history:
        return message

    words = message.lower().split()
    word_set = {w.strip(".,?!;:") for w in words}

    is_short = len(words) <= 6
    is_vague = bool(word_set & _VAGUE_WORDS)    # pronouns make subject ambiguous regardless
    has_reg_content = bool(word_set & _REGULATORY_WORDS)

    # Expand if: pronoun makes subject ambiguous, OR: short + no regulatory content
    if is_vague or (is_short and not has_reg_content):
        # Find the most recent user message for context
        prev = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"),
            None,
        )
        if prev and prev != message:
            # Use at most 12 words from the previous message to keep query focused
            prev_words = prev.split()[:12]
            return " ".join(prev_words) + " — " + message

    return message


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream an answer to *req.message* as Server-Sent Events.

    Event format::

        data: {"token": "some text", "done": false}
        data: {"token": "", "done": true, "sources": [...], "message_id": "..."}
    """
    message_id = str(uuid4())
    history = _get_history(req.conversation_id)

    async def event_stream():
        # ── 0. Classify intent (rule-based, ~0ms) ─────────────────────────
        intent = classify_intent(req.message)

        # ── 1. Signal activity so UI responds immediately ─────────────────
        yield _sse({"token": "", "done": False, "searching": True, "intent": intent})

        # ── 2. Retrieve relevant regulations (skipped for conversational) ─
        if intent == "conversational":
            sources: list[Source] = []
            retrieval_ms = 0.0
        else:
            # Expand vague follow-ups with prior context for better retrieval
            retrieval_query = _retrieval_query(req.message, history)
            sources, retrieval_ms = await retrieve(
                query=retrieval_query,
                top_k=req.top_k,
                filters=req.filters or None,
            )

        # ── 3. Stream LLM response ─────────────────────────────────────────
        from api.config import get_settings
        mode = "live" if get_settings().llm_ready else "stub"
        full_answer = []

        try:
            async for token in generate_stream(req.message, sources, history, intent=intent):
                full_answer.append(token)
                yield _sse({"token": token, "done": False})
        except Exception as exc:
            logger.error("Stream error: %s", exc, exc_info=True)
            yield _sse({"token": "\n\n[An error occurred while generating the response. Please try again.]", "done": False})

        # ── 4. Verify citations (skip for conversational — no sources) ────────
        answer_text = "".join(full_answer)
        verification: dict = {}
        if _verifier and sources and answer_text:
            try:
                verification = _verifier.verify(
                    answer_text,
                    [{"citation": s.citation} for s in sources],
                )
                n_verified = len(verification.get("verified", []))
                unverified = verification.get("unverified", [])
                if unverified:
                    extra = (
                        f"\n\n---\n*⚠️ {len(unverified)} citation(s) in this response were not found "
                        f"in the retrieved sources: {', '.join(unverified[:3])}.*"
                    )
                    yield _sse({"token": extra, "done": False})
                    full_answer.append(extra)
                    answer_text = answer_text + extra
                elif n_verified:
                    extra = f"\n\n---\n*📋 {n_verified} citation(s) verified against retrieved sources.*"
                    yield _sse({"token": extra, "done": False})
                    full_answer.append(extra)
                    answer_text = answer_text + extra
            except Exception as exc:
                logger.warning("Citation verification error: %s", exc)

        # ── 5. Final event with metadata ───────────────────────────────────
        yield _sse({
            "token": "",
            "done": True,
            "sources": [s.model_dump() for s in sources],
            "conversation_id": req.conversation_id,
            "message_id": message_id,
            "retrieval_ms": retrieval_ms,
            "mode": mode,
            "intent": intent,
            "verification": {
                "verified": verification.get("verified", []),
                "unverified": verification.get("unverified", []),
                "hallucination_risk": verification.get("hallucination_risk", False),
            } if verification else None,
        })

        # ── 6. Update conversation history ────────────────────────────────
        _append_history(req.conversation_id, req.message, answer_text)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """Direct vector search without LLM generation — useful for debugging."""
    sources, elapsed = await retrieve(req.query, top_k=req.top_k, filters=req.filters or None)
    return SearchResponse(results=sources, query=req.query, retrieval_ms=elapsed)
