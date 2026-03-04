"""pipeline/citation_verifier.py — Post-generation citation hallucination check.

After the LLM generates a response, this module:
1. Extracts every COMAR citation mentioned in the response.
2. Checks each against the set of citations that were actually retrieved.
3. Flags any citation that appears in the response but NOT in retrieved context
   as a potential hallucination.
4. Optionally prepends a warning and appends a verification summary footer.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Matches "COMAR 15.05.01.06" (4-part) or "COMAR 15.05.01" (3-part)
_CITATION_RE = re.compile(
    r"COMAR\s+\d{1,2}\.\d{2}\.\d{2}(?:\.\d{2})?",
    re.IGNORECASE,
)


class CitationVerifier:
    """Verify that LLM-generated citations are grounded in retrieved context."""

    # ── Public API ────────────────────────────────────────────────────────────

    def verify(
        self,
        response: str,
        retrieved_chunks: list[dict],
    ) -> dict[str, object]:
        """Extract and validate all COMAR citations in *response*.

        A citation is considered **verified** when it appears in the
        ``citation`` field of at least one retrieved chunk.  Any citation
        absent from the retrieved set is flagged as potentially hallucinated.

        Args:
            response: The LLM-generated answer text.
            retrieved_chunks: Chunks returned by the retrieval pipeline.
                              Each dict should contain a ``"citation"`` key
                              (directly or under ``"metadata"``).

        Returns:
            Dict with keys:

            ``verified``          — list of citation strings that match retrieved chunks
            ``unverified``        — list of citation strings not in retrieved chunks
            ``hallucination_risk`` — True when *unverified* is non-empty
        """
        # Normalise: COMAR citations to upper-case for comparison
        found = [m.upper() for m in _CITATION_RE.findall(response)]
        found_unique = list(dict.fromkeys(found))  # deduplicate, preserve order

        # Build valid citation set from retrieved chunks
        valid: set[str] = set()
        for chunk in retrieved_chunks:
            # Citation may be at top level or under "metadata"
            cit = chunk.get("citation") or chunk.get("metadata", {}).get("citation", "")
            if cit:
                valid.add(cit.upper())
            # Also accept 3-part prefix match: "COMAR 15.05.01" covers "COMAR 15.05.01.06"
            parts = cit.upper().rsplit(".", 1)
            if len(parts) == 2:
                valid.add(parts[0])  # e.g. "COMAR 15.05.01"

        def _is_verified(cit: str) -> bool:
            if cit in valid:
                return True
            # Also accept when the 3-part prefix of the found citation is in valid.
            # Handles cases like found "COMAR 26.08.04.02" matching retrieved
            # "COMAR 26.08.04.02-3" (whose 3-part prefix "COMAR 26.08.04" is in valid).
            prefix = cit.rsplit(".", 1)[0]
            return prefix in valid

        verified = [c for c in found_unique if _is_verified(c)]
        unverified = [c for c in found_unique if not _is_verified(c)]
        hallucination_risk = len(unverified) > 0

        logger.debug(
            "CitationVerifier: %d verified, %d unverified (hallucination_risk=%s)",
            len(verified),
            len(unverified),
            hallucination_risk,
        )

        return {
            "verified": verified,
            "unverified": unverified,
            "hallucination_risk": hallucination_risk,
        }

    def add_verification_footer(
        self,
        response: str,
        verification: dict[str, object],
    ) -> str:
        """Augment *response* with citation verification metadata.

        Prepends a hallucination warning when ``verification["hallucination_risk"]``
        is ``True``, and appends a one-line citation summary at the bottom.

        Args:
            response: Original LLM response text.
            verification: Output of :meth:`verify`.

        Returns:
            Modified response string.
        """
        if verification.get("hallucination_risk"):
            unverified = verification.get("unverified", [])
            warning = (
                "\u26a0\ufe0f **WARNING**: The following citations could not be "
                f"verified against retrieved context: {', '.join(unverified)}. "
                "Please verify these independently before relying on them.\n\n"
            )
            response = warning + response

        v_count = len(verification.get("verified", []))
        u_count = len(verification.get("unverified", []))
        footer = (
            f"\n\n---\n"
            f"\U0001f4cb *Citation verification: {v_count} verified"
            + (f", {u_count} unverified" if u_count else "")
            + ".*"
        )
        return response + footer
