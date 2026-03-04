"""api/services/intent.py — Zero-latency rule-based intent classifier.

Classifies a user message into one of seven intent labels without any LLM
call — pure keyword + regex matching, ~0ms overhead per request.

Intent labels
─────────────
conversational  — greetings, thanks, meta-questions ("what can you do?")
citation_lookup — explicit COMAR XX.XX.XX reference in query
definition      — "what is X", "define X", "meaning of X"
compliance      — requirements, permits, licensing, "must I"
overview        — broad summaries, "what does Title X cover"
enforcement     — penalties, fines, violations, sanctions
general         — any other regulatory query (fallback)
"""
from __future__ import annotations

import re

_CITATION_RE = re.compile(r"COMAR\s+\d{1,2}\.\d{2}\.\d{2}", re.IGNORECASE)

# Single-word conversational triggers — matched as whole words only (not substrings).
# E.g. "hi" should NOT match inside "this" or "vehicle".
_CONVERSATIONAL_WORDS: frozenset[str] = frozenset({
    "hi", "hello", "hey", "hiya", "howdy",
    "thanks", "thx", "ty", "cheers",
    "bye", "goodbye", "sup",
    "help",
})

# Multi-word conversational phrases — substring-matched against the full lowercased message.
_CONVERSATIONAL_PHRASES: frozenset[str] = frozenset({
    "good morning", "good afternoon", "good evening", "good night",
    "thank you", "thank you so",
    "see you", "take care",
    "how are you", "how r u", "what's up", "whats up",
    "what can you do", "what do you do", "what can you help with",
    "who are you", "what are you", "what is this",
    "tell me about yourself", "tell me about you",
    "can you help", "help me",
    "how does this work", "how do you work",
    "what topics do you cover", "what can i ask",
    "got it", "understood",
})

# Regulatory markers — if any are present the message is NOT conversational
# even if it matches a conversational phrase (e.g. "help me with COMAR").
_REGULATORY_MARKERS: tuple[str, ...] = (
    "comar", "regulation", "title", "section", "subtitle",
    "chapter", "code of maryland", "maryland regulation",
)

# Ordered keyword lists for regulatory intent classification.
# IMPORTANT: Order matters — first match wins.
# More specific intents (enforcement, definition) come before broader ones (compliance, overview).
_INTENT_KEYWORDS: dict[str, list[str]] = {
    "enforcement": [
        "penalty", "penalties", " fine ", "fines", "violat",
        "sanction", "enforcement", "consequence",
        "punishment", "revoke", "suspend", "criminal", "civil penalty",
        "what happens if", "disciplinary",
    ],
    "definition": [
        "what is", "define ", "definition of", "definition:",
        "meaning of", "what do you mean by",
        "how is defined", "what does it mean", "clarify", "explained",
    ],
    "compliance": [
        "requirement", "require ", "must ", "need to", "do i need",
        "am i required", "permit", "license", "certification", "certif",
        "comply", "compliance", "allowed to", "can i ", "may i ",
        "prohibited", "restriction", "regulated", "authorized",
        "application for", "apply for", "registration",
    ],
    "overview": [
        "overview", "summary", "summarize", "general overview",
        "tell me about title", "what does title", "what is comar title",
        "explain title", "what topics", "broad", "introduction to",
        "what is covered", "all regulations", "scope of",
    ],
}


def classify(message: str) -> str:
    """Return the intent label for *message*.

    Classification order:
      1. conversational — short message, no regulatory markers, matches known phrase
      2. citation_lookup — explicit COMAR citation in text (regex)
      3. definition / compliance / overview / enforcement — keyword matching
      4. general — fallback
    """
    msg = message.strip()
    lower = msg.lower()

    # ── 1. Conversational ──────────────────────────────────────────────────
    if len(msg.split()) <= 12:
        has_reg = any(marker in lower for marker in _REGULATORY_MARKERS)
        if not has_reg:
            words = set(lower.split())
            # Single-word triggers: exact word match (avoids "hi" inside "this")
            single_match = bool(words & _CONVERSATIONAL_WORDS)
            # Multi-word triggers: substring match (space-separated phrases are safe)
            multi_match = any(phrase in lower for phrase in _CONVERSATIONAL_PHRASES)
            if single_match or multi_match:
                return "conversational"

    # ── 2. Direct citation lookup ──────────────────────────────────────────
    if _CITATION_RE.search(msg):
        return "citation_lookup"

    # ── 3. Keyword-based regulatory intents (first match wins) ────────────
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return intent

    # ── 4. Fallback ───────────────────────────────────────────────────────
    return "general"
