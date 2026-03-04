"""pipeline/router.py — Query classification and citation-lookup detection.

Classifies incoming queries into one of five COMAR-relevant categories so
the pipeline can choose the optimal retrieval path.  When the LLM is
unavailable, a keyword-based heuristic provides a reasonable fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pipeline.prompts import ROUTER_PROMPT

logger = logging.getLogger(__name__)

# Regex: matches "COMAR 15.05.01.06" — with or without the final segment
_CITATION_RE = re.compile(
    r"\bCOMAR\s+\d{1,2}\.\d{2}\.\d{2}(?:\.\d{2})?\b",
    re.IGNORECASE,
)

_VALID_CATEGORIES = frozenset(
    {"citation_lookup", "definition", "compliance", "cross_ref", "procedural"}
)


class QueryRouter:
    """Classify queries and detect direct citation lookups.

    Args:
        llm: An instantiated LangChain chat model (e.g. ChatAnthropic).
             Pass ``None`` to use keyword-only classification.
    """

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    # ── Public API ────────────────────────────────────────────────────────────

    def is_citation_lookup(self, query: str) -> bool:
        """Return True if *query* contains a COMAR citation pattern.

        Matches patterns like ``COMAR 15.05.01.06`` or ``COMAR 26.08.02``.
        """
        return bool(_CITATION_RE.search(query))

    def classify(self, query: str) -> str:
        """Return the category name for *query* (one of five strings).

        Tries the LLM first; falls back to keyword heuristics on failure.
        """
        if self.llm is not None:
            try:
                prompt = ROUTER_PROMPT.format(query=query)
                response = self.llm.invoke(prompt)
                category = (
                    response.content.strip().lower()
                    if hasattr(response, "content")
                    else str(response).strip().lower()
                )
                if category in _VALID_CATEGORIES:
                    logger.debug("Router (LLM): %s → %s", query[:60], category)
                    return category
            except Exception as exc:
                logger.warning("Router LLM call failed (%s); using heuristic", exc)

        return self._heuristic_classify(query)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _heuristic_classify(self, query: str) -> str:
        """Keyword-based fallback classifier."""
        q = query.lower()

        if self.is_citation_lookup(query):
            return "citation_lookup"

        if any(w in q for w in ("definition", "define", "what is", "what does", "mean", "term")):
            return "definition"

        if any(w in q for w in ("how to", "how do i", "apply for", "file", "submit", "process", "procedure", "steps")):
            return "procedural"

        if any(w in q for w in ("cross-reference", "cross reference", "referenced by", "links to", "related to")):
            return "cross_ref"

        # Default: compliance (permits, requirements, obligations)
        return "compliance"
