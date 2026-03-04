"""api/services/llm.py — LLM response generation via DeepSeek (OpenAI-compatible).

Streams token-by-token responses.  Falls back to a stub answer when no API
key is configured so the frontend streaming path is always exercised.

Provider priority: deepseek → anthropic → openai → stub.
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import AsyncIterator

from api.config import get_settings
from api.models import Source

logger = logging.getLogger(__name__)

_LLM_TIMEOUT = 60  # seconds — prevents indefinite hangs on DeepSeek


@lru_cache(maxsize=1)
def _get_openai_client(api_key: str, base_url: str | None = None):
    """Singleton AsyncOpenAI client per (api_key, base_url) combo."""
    from openai import AsyncOpenAI
    kwargs: dict = {"api_key": api_key, "timeout": _LLM_TIMEOUT}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


# ── Shared grounding rules appended to all regulatory prompts ─────────────────
_BASE_RULES = """
Grounding rules:
- Begin directly with the substance — never open with "Based on the retrieved sources", \
"According to the provided regulatory text", or similar boilerplate.
- Ground every factual claim in the retrieved sources above. Cite each claim with its \
exact COMAR citation (e.g., COMAR 15.05.01.06).
- Quote key definitions or requirements precisely, then explain in plain language.
- If information is not in the retrieved sources, acknowledge it briefly and pivot to \
what is available.
- Close with: "Always verify current requirements at regs.maryland.gov."
- Append once at the very end: "*For informational research only — not legal advice.*"
"""

# ── Intent-keyed system prompts ───────────────────────────────────────────────
_PROMPTS: dict[str, str] = {

    "conversational": """\
You are COMAR Assistant, a helpful AI research tool for the Code of Maryland Regulations.
Be concise and friendly. If asked what you can do:
- Answer questions about COMAR Title 15 (Agriculture) and Title 26 (Environment)
- Look up regulations, definitions, and compliance requirements with direct citations
- Handle multi-turn research conversations
Do not fabricate regulatory content.""",

    "citation_lookup": """\
You are COMAR Assistant, an expert on the Code of Maryland Regulations.
The user has asked about a specific COMAR citation.
Lead with the retrieved regulation text. Explain its practical meaning in plain language. \
Note any key definitions, exceptions, or cross-references in the source.
""" + _BASE_RULES,

    "definition": """\
You are COMAR Assistant, an expert on the Code of Maryland Regulations.
The user is asking for a regulatory definition.
Quote the statutory definition exactly, then explain it in plain language. \
Note the scope (which title/chapter it governs). \
If multiple related definitions appear, present each one clearly.
""" + _BASE_RULES,

    "compliance": """\
You are COMAR Assistant, an expert on Maryland regulatory compliance.
Answer the compliance question in a structured way: cover what is required, \
who it applies to, and practical steps to comply. Use headers and bullet points \
where they aid clarity. Cite every requirement with its COMAR citation.
""" + _BASE_RULES,

    "overview": """\
You are COMAR Assistant, an expert on the Code of Maryland Regulations.
The user wants a broad overview. Synthesize across all retrieved sources using \
headers for major topics. Cover key regulations, definitions, and compliance themes. \
Keep the response structured and scannable.
""" + _BASE_RULES,

    "enforcement": """\
You are COMAR Assistant, an expert on Maryland regulatory enforcement.
Answer the enforcement question clearly, covering penalty provisions, enforcement \
mechanisms, and any mitigating factors present in the retrieved sources. \
Use headers where helpful. Cite each provision with its exact COMAR citation.
""" + _BASE_RULES,

    "general": """\
You are COMAR Assistant, a knowledgeable expert on the Code of Maryland Regulations (COMAR), \
specialising in Title 15 (Agriculture) and Title 26 (Environment).
Speak as a regulatory expert, not as a system reading documents. \
Answer directly. Use headers and bullet points for structured topics. \
Keep the tone expert but accessible.
""" + _BASE_RULES,
}


def _strip_breadcrumb(text: str) -> str:
    """Remove the leading breadcrumb line from chunk_text.

    Every chunk starts with a long hierarchy path ending at the regulation name,
    e.g. 'Title 15 — ... > Regulation .01 Definitions.\\n\\n(1) "Applicant" means...'
    Since SOURCE headers already carry the citation, this prefix is redundant noise.
    """
    if text.startswith("Title "):
        idx = text.find("\n\n")
        if idx != -1:
            return text[idx + 2:]
    return text


def _build_context(sources: list[Source]) -> str:
    if not sources:
        return "No regulatory sources retrieved."
    parts = []
    for i, s in enumerate(sources, 1):
        header = f"SOURCE {i}: {s.citation}"
        if s.regulation_name:
            header += f" — {s.regulation_name}"
        if s.effective_date:
            header += f" (effective {s.effective_date})"
        body = _strip_breadcrumb(s.text_snippet)
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


async def generate_stream(
    message: str,
    sources: list[Source],
    conversation_history: list[dict] | None = None,
    intent: str = "general",
) -> AsyncIterator[str]:
    """Async generator that yields text tokens.

    Falls back to a character-by-character stub stream when no LLM is
    configured so the frontend streaming code path is always exercised.
    """
    settings = get_settings()

    if not settings.llm_ready:
        async for token in _stub_stream(message, sources):
            yield token
        return

    ph = "your_key_here"
    provider = settings.llm_provider.lower()

    if provider == "deepseek" and settings.deepseek_api_key != ph:
        async for token in _openai_compatible_stream(
            message, sources, conversation_history or [],
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            intent=intent,
        ):
            yield token
        return

    if provider == "anthropic" and settings.anthropic_api_key != ph:
        async for token in _anthropic_stream(
            message, sources, conversation_history or [], intent=intent,
        ):
            yield token
        return

    if provider == "openai" and settings.openai_api_key != ph:
        async for token in _openai_compatible_stream(
            message, sources, conversation_history or [],
            api_key=settings.openai_api_key,
            base_url=None,
            model=settings.llm_model,
            intent=intent,
        ):
            yield token
        return

    async for token in _stub_stream(message, sources):
        yield token


async def _openai_compatible_stream(
    message: str,
    sources: list[Source],
    history: list[dict],
    api_key: str,
    base_url: str | None,
    model: str,
    intent: str = "general",
) -> AsyncIterator[str]:
    """Stream tokens via any OpenAI-compatible API (DeepSeek, OpenAI, etc.)."""
    try:
        client = _get_openai_client(api_key, base_url)
        system_prompt = _PROMPTS.get(intent, _PROMPTS["general"])

        # Conversational: no regulatory context block needed
        if intent == "conversational" or not sources:
            user_content = message
        else:
            context = _build_context(sources)
            user_content = f"Regulatory context:\n{context}\n\nQuestion: {message}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            temperature=0.1,
            stream=True,
        )
        async for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text

    except Exception as exc:
        logger.error("LLM streaming error (%s): %s", model, exc)
        async for token in _stub_stream(message, sources):
            yield token


async def _anthropic_stream(
    message: str,
    sources: list[Source],
    history: list[dict],
    intent: str = "general",
) -> AsyncIterator[str]:
    """Stream tokens from Claude via the Anthropic SDK (fallback)."""
    try:
        import anthropic

        settings = get_settings()
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        system_prompt = _PROMPTS.get(intent, _PROMPTS["general"])

        if intent == "conversational" or not sources:
            user_content = message
        else:
            context = _build_context(sources)
            user_content = f"Regulatory context:\n{context}\n\nQuestion: {message}"

        messages = list(history)
        messages.append({"role": "user", "content": user_content})

        async with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    except Exception as exc:
        logger.error("Anthropic streaming error: %s", exc)
        async for token in _stub_stream(message, sources):
            yield token


async def _stub_stream(message: str, sources: list[Source]) -> AsyncIterator[str]:
    """Simulate a streaming response for demo / pre-configuration use."""
    citations = ", ".join(s.citation for s in sources[:3]) if sources else "various COMAR regulations"

    answer = (
        f"Based on the retrieved regulatory sources ({citations}), here is what "
        f"COMAR provides in response to your question about **{message[:60]}{'...' if len(message) > 60 else ''}**:\n\n"
        "The Maryland regulations establish specific requirements that govern this area. "
        "The relevant provisions define the applicable standards, procedures, and enforcement "
        "mechanisms that regulated entities must follow.\n\n"
        "> **Note:** COMAR Assistant is in stub mode. "
        "Add your `DEEPSEEK_API_KEY` to `.env` to enable full AI-powered responses."
    )

    words = answer.split(" ")
    for i, word in enumerate(words):
        await asyncio.sleep(0.03)
        yield word + (" " if i < len(words) - 1 else "")
