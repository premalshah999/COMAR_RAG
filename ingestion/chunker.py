"""chunker.py — Convert regulation dicts into embeddable chunk dicts.

Produces THREE chunk types from each parsed regulation:

PRIMARY
    One chunk per regulation containing the full text, prefixed with a
    human-readable breadcrumb so retrieval context is always self-contained.

SUBSECTION
    When a regulation's text exceeds ``SUBSECTION_TOKEN_THRESHOLD`` (600)
    tokens the text is split at paragraph markers (A., B., (1), (2) …) into
    child chunks. Each subsection records the parent regulation's chunk_id.

DEFINITION
    Every ``.01 Definitions`` regulation also feeds a *definitions lookup*:
    a dict mapping each defined term (lower-cased) to its definition text and
    source citation. These are NOT added to the chunk list a second time, but
    returned separately for use by the graph builder and query router.

Returned chunk dicts carry all original metadata PLUS:

    chunk_text   : str     — the text that will be embedded
    chunk_type   : str     — "regulation" | "definition" | "subsection"
    parent_id    : str | None — parent regulation chunk_id (subsections only)
    word_count   : int
    token_count  : int

Usage::

    from ingestion.chunker import create_chunks

    chunks, definitions = create_chunks(regulations)
    # chunks: list[dict], definitions: {term: {definition_text, source_citation}}
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SUBSECTION_TOKEN_THRESHOLD: int = 600
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Paragraph markers used to split regulation text into subsections.
# Pattern matches labels at the START of a word boundary preceded by whitespace
# or start-of-string:  A.  B.  (1)  (2)  (a)  (ii) etc.
_SUBSECTION_SPLIT_RE = re.compile(
    r"(?<!\w)"                          # not preceded by a word char
    r"(?="                              # lookahead (keep the marker)
    r"(?:[A-Z]\.\s)"                    # A.  B.  C.
    r"|"
    r"(?:\(\d+\)\s)"                    # (1)  (2)
    r"|"
    r"(?:\([a-z]+\)\s)"                 # (a)  (b)  (ii)
    r")"
)

# Regex to extract individual defined terms from a Definitions regulation text.
# Looks for quoted terms like  "Secretary" means …  or  "Secretary" has the meaning …
_DEFINED_TERM_RE = re.compile(
    r'"([^"]{1,80})"'                   # quoted term, max 80 chars
    r"\s+"
    r"(?:means|has the meaning|refers to|is defined as)"
    r"([^""\n]{5,600})",                # definition body, up to 600 chars
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _token_count(text: str) -> int:
    """Return the number of cl100k_base tokens in *text*."""
    return len(_TOKENIZER.encode(text))


def _word_count(text: str) -> int:
    return len(text.split())


def _breadcrumb(reg: dict[str, Any]) -> str:
    """Build a human-readable breadcrumb prefix for the chunk text.

    Example::

        Title 15 — MARYLAND DEPARTMENT OF AGRICULTURE >
        Subtitle 01 — OFFICE OF THE SECRETARY >
        Chapter 01 — Procedural Regulations >
        Regulation .01 Definitions.
    """
    return (
        f"Title {reg['title_num']} — {reg['title_name']} > "
        f"Subtitle {reg['subtitle_num']} — {reg['subtitle_name']} > "
        f"Chapter {reg['chapter_num']} — {reg['chapter_name']} > "
        f"Regulation .{reg['regulation_num']} {reg['regulation_name']}"
    )


def _make_base_chunk(reg: dict[str, Any]) -> dict[str, Any]:
    """Copy all metadata fields from a regulation dict into a new chunk dict."""
    chunk = {k: v for k, v in reg.items() if k != "text"}
    return chunk


def _split_into_subsections(text: str) -> list[str]:
    """Split regulation text at paragraph boundary markers.

    Returns a list of non-empty string segments.  The markers themselves are
    preserved at the start of each segment.
    """
    parts = _SUBSECTION_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── Definition extraction ──────────────────────────────────────────────────────


def _extract_definitions(
    reg: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Parse individual term→definition pairs from a .01 Definitions regulation.

    Args:
        reg: A regulation dict whose ``chunk_type`` is ``"definition"``.

    Returns:
        Mapping ``{term_lower: {"definition_text": ..., "source_citation": ...}}``.
    """
    lookup: dict[str, dict[str, str]] = {}
    text = reg.get("text", "")
    citation = reg.get("citation", "")

    for match in _DEFINED_TERM_RE.finditer(text):
        term = match.group(1).strip()
        definition_body = match.group(2).strip()
        # Trim at the next quoted term or end-of-sentence to avoid run-on defs
        definition_body = re.split(r'"\w', definition_body)[0].strip()
        if term and definition_body:
            lookup[term.lower()] = {
                "definition_text": f'"{term}" {definition_body}',
                "source_citation": citation,
            }

    if lookup:
        logger.debug(
            "Extracted %d definitions from %s", len(lookup), citation
        )
    return lookup


# ── Primary chunk factory ─────────────────────────────────────────────────────


def _make_primary_chunk(reg: dict[str, Any]) -> dict[str, Any]:
    """Build the PRIMARY chunk for a regulation."""
    breadcrumb = _breadcrumb(reg)
    chunk_text = f"{breadcrumb}\n\n{reg['text']}"
    chunk = _make_base_chunk(reg)
    chunk.update(
        {
            "chunk_text": chunk_text,
            # chunk_type already set (regulation | definition), keep as-is
            "parent_id": None,
            "word_count": _word_count(chunk_text),
            "token_count": _token_count(chunk_text),
        }
    )
    return chunk


# ── Subsection chunk factory ──────────────────────────────────────────────────


def _make_subsection_chunks(
    reg: dict[str, Any], primary_token_count: int
) -> list[dict[str, Any]]:
    """Split an over-long regulation into subsection chunks.

    Only called when ``primary_token_count > SUBSECTION_TOKEN_THRESHOLD``.

    Args:
        reg: The regulation dict.
        primary_token_count: Token count of the already-built primary chunk.

    Returns:
        List of subsection chunk dicts (may be empty if splitting fails).
    """
    segments = _split_into_subsections(reg["text"])
    if len(segments) <= 1:
        # Cannot split further — leave as a single primary chunk
        return []

    breadcrumb = _breadcrumb(reg)
    parent_id = reg["chunk_id"]
    subsections: list[dict[str, Any]] = []

    for idx, segment in enumerate(segments):
        # Give each subsection a stable id: COMAR.15.01.01.02.sub.0
        sub_id = f"{parent_id}.sub.{idx}"
        chunk_text = f"{breadcrumb}\n\n{segment}"
        chunk = _make_base_chunk(reg)
        chunk.update(
            {
                "chunk_id": sub_id,
                "chunk_text": chunk_text,
                "chunk_type": "subsection",
                "parent_id": parent_id,
                "word_count": _word_count(chunk_text),
                "token_count": _token_count(chunk_text),
            }
        )
        subsections.append(chunk)

    logger.debug(
        "Split %s (%d tokens) → %d subsections",
        reg["chunk_id"],
        primary_token_count,
        len(subsections),
    )
    return subsections


# ── Public API ────────────────────────────────────────────────────────────────


def create_chunks(
    regulations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Convert a list of regulation dicts into embeddable chunks.

    For each regulation:

    1. Always produces a **primary** chunk (full regulation text with breadcrumb).
    2. If the primary chunk exceeds :data:`SUBSECTION_TOKEN_THRESHOLD` tokens,
       also produces **subsection** child chunks split on paragraph markers.
    3. For ``.01`` definition regulations, extracts individual term→definition
       pairs into the returned *definitions lookup*.

    Args:
        regulations: Output of :func:`~ingestion.xml_parser.parse_comar_xml`.

    Returns:
        A 2-tuple ``(chunks, definitions)`` where:

        - ``chunks`` is a flat list of chunk dicts ready for embedding.
        - ``definitions`` maps lower-cased term strings to
          ``{"definition_text": str, "source_citation": str}`` dicts.
    """
    all_chunks: list[dict[str, Any]] = []
    definitions_lookup: dict[str, dict[str, str]] = {}

    long_reg_count = 0
    subsection_total = 0

    for reg in regulations:
        # ── Primary chunk ─────────────────────────────────────────────────
        primary = _make_primary_chunk(reg)
        all_chunks.append(primary)

        # ── Subsection chunks ─────────────────────────────────────────────
        if primary["token_count"] > SUBSECTION_TOKEN_THRESHOLD:
            long_reg_count += 1
            subs = _make_subsection_chunks(reg, primary["token_count"])
            all_chunks.extend(subs)
            subsection_total += len(subs)

        # ── Definition extraction ──────────────────────────────────────────
        if reg.get("chunk_type") == "definition":
            defs = _extract_definitions(reg)
            definitions_lookup.update(defs)

    logger.info(
        "Chunking complete: %d regulations → %d primary + %d subsection = %d total chunks",
        len(regulations),
        len(regulations),
        subsection_total,
        len(all_chunks),
    )
    logger.info(
        "Long regulations split: %d  |  Definitions extracted: %d",
        long_reg_count,
        len(definitions_lookup),
    )

    return all_chunks, definitions_lookup
