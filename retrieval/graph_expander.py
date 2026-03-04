"""retrieval/graph_expander.py — Expand retrieval candidates via the knowledge graph.

Given a set of chunk_ids already retrieved by vector search, this module walks
the NetworkX knowledge graph to pull in:

* **Cross-referenced regulations** — outgoing REFERENCES edges from each chunk.
* **Definition regulations** — the ``.01 Definitions`` chunk for every chapter
  represented in the candidate set (ensures relevant terminology is always
  included in context).

The expanded set is deduplicated and returned as a list of additional chunk_ids
to fetch from Qdrant and pass to the reranker.

Usage::

    from retrieval.graph_expander import GraphExpander

    expander = GraphExpander()                         # loads data/comar_graph.pkl
    extra_ids = expander.expand(["COMAR.15.05.01.06"])
    path      = expander.get_context_path("COMAR.15.05.01.06")
    # → "Title 15 MARYLAND DEPARTMENT OF AGRICULTURE > Subtitle 05 ..."
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx

from ingestion.graph_builder import (
    get_chapter_for_regulation,
    get_cross_refs,
    get_definitions_for_chapter,
    load_graph,
)

logger = logging.getLogger(__name__)

_DEFAULT_GRAPH_PATH = Path("./data/comar_graph.pkl")


class GraphExpander:
    """Expands retrieval candidates using the COMAR knowledge graph.

    Args:
        graph_path: Path to the pickled :class:`networkx.DiGraph`.  Defaults
                    to ``./data/comar_graph.pkl``.
    """

    def __init__(self, graph_path: Path = _DEFAULT_GRAPH_PATH) -> None:
        self.graph: nx.DiGraph = load_graph(Path(graph_path))

    # ── Public API ────────────────────────────────────────────────────────────

    def expand(self, chunk_ids: list[str]) -> list[str]:
        """Return additional chunk_ids suggested by graph traversal.

        For each chunk_id in *chunk_ids*:
        1. Follow outgoing ``REFERENCES`` edges → add cited regulation ids.
        2. Find the parent chapter node → add its ``.01 Definitions`` chunk_id.

        Subsection ids (containing ``.sub.``) are resolved to their parent
        regulation before graph lookup.

        Args:
            chunk_ids: Chunk ids from the initial vector retrieval.

        Returns:
            Deduplicated list of *additional* chunk_ids not already in the
            input set.
        """
        input_set = set(chunk_ids)
        additional: set[str] = set()

        for chunk_id in chunk_ids:
            # Resolve subsection → parent regulation for graph lookups
            base_id = chunk_id.split(".sub.")[0]

            # ── Cross-references ──────────────────────────────────────────
            for ref_id in get_cross_refs(base_id, self.graph):
                additional.add(ref_id)

            # ── Definition chunk for the chapter ──────────────────────────
            chapter_id = get_chapter_for_regulation(base_id, self.graph)
            if chapter_id:
                for def_id in get_definitions_for_chapter(chapter_id, self.graph):
                    additional.add(def_id)

        new_ids = [cid for cid in additional if cid not in input_set]
        logger.debug(
            "GraphExpander: %d inputs → %d expanded ids added",
            len(chunk_ids),
            len(new_ids),
        )
        return new_ids

    def get_context_path(self, chunk_id: str) -> str:
        """Build a human-readable breadcrumb for *chunk_id*.

        Looks up node attributes in the graph to produce names like::

            Title 15 MARYLAND DEPARTMENT OF AGRICULTURE >
            Subtitle 05 PESTICIDE REGULATION >
            Chapter 01 Definitions >
            Regulation .06

        Falls back to parsing the chunk_id string if the node is not in the
        graph (e.g. for subsection ids or graph gaps).

        Args:
            chunk_id: Regulation or subsection chunk_id.

        Returns:
            A ``" > "``-separated breadcrumb string.
        """
        base_id = chunk_id.split(".sub.")[0]

        if self.graph.has_node(base_id):
            node = self.graph.nodes[base_id]
            t = node.get("title_num", "")
            sub = node.get("subtitle_num", "")
            ch = node.get("chapter_num", "")
            reg = node.get("regulation_num", "")

            tid = f"TITLE.{t}"
            sid = f"SUBTITLE.{t}.{sub}"
            cid = f"CHAPTER.{t}.{sub}.{ch}"

            title_name = (
                self.graph.nodes[tid].get("title_name", "")
                if self.graph.has_node(tid)
                else ""
            )
            subtitle_name = (
                self.graph.nodes[sid].get("subtitle_name", "")
                if self.graph.has_node(sid)
                else ""
            )
            chapter_name = (
                self.graph.nodes[cid].get("chapter_name", "")
                if self.graph.has_node(cid)
                else ""
            )

            parts = [
                f"Title {t} {title_name}".strip(),
                f"Subtitle {sub} {subtitle_name}".strip(),
                f"Chapter {ch} {chapter_name}".strip(),
                f"Regulation .{reg}",
            ]
            return " > ".join(parts)

        # Fallback: derive from the chunk_id string
        # "COMAR.15.05.01.06" → ["15","05","01","06"]
        raw = base_id.replace("COMAR.", "")
        parts = raw.split(".")
        if len(parts) >= 4:
            return (
                f"Title {parts[0]} > Subtitle {parts[1]} > "
                f"Chapter {parts[2]} > Regulation .{parts[3]}"
            )
        return base_id
