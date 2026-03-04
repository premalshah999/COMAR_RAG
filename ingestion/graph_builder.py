"""graph_builder.py — Build a NetworkX knowledge graph from COMAR regulations.

Graph structure
---------------
Nodes
    Every unique entity in the COMAR hierarchy has a node keyed by its ID
    string.  Four node types:

    * ``title``       — e.g. ``"TITLE.15"``
    * ``subtitle``    — e.g. ``"SUBTITLE.15.01"``
    * ``chapter``     — e.g. ``"CHAPTER.15.01.01"``
    * ``regulation``  — e.g. ``"COMAR.15.01.01.01"``

    Each node carries all metadata available from the regulation dicts.

Edges (directed)
    * ``CONTAINS``     — parent → child in the hierarchy
    * ``REFERENCES``   — regulation A → regulation B when A.cross_refs ∋ B.citation
    * ``DEFINES``      — definition regulation → parent chapter node

The graph is serialised to ``./data/comar_graph.pkl`` with :func:`pickle`.

Public API
----------
.. code-block:: python

    from ingestion.graph_builder import build_knowledge_graph, get_cross_refs

    graph = build_knowledge_graph(regulations)
    refs  = get_cross_refs("COMAR.15.01.01.03", graph)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_GRAPH_PATH = Path("./data/comar_graph.pkl")


# ── ID helpers ─────────────────────────────────────────────────────────────────


def _title_id(title_num: str) -> str:
    return f"TITLE.{title_num}"


def _subtitle_id(title_num: str, subtitle_num: str) -> str:
    return f"SUBTITLE.{title_num}.{subtitle_num}"


def _chapter_id(title_num: str, subtitle_num: str, chapter_num: str) -> str:
    return f"CHAPTER.{title_num}.{subtitle_num}.{chapter_num}"


# ── Build ──────────────────────────────────────────────────────────────────────


def build_knowledge_graph(
    regulations: list[dict[str, Any]],
    save_path: Path | None = DEFAULT_GRAPH_PATH,
) -> nx.DiGraph:
    """Build a directed knowledge graph from parsed COMAR regulations.

    Iterates through every regulation to:

    1. Ensure title / subtitle / chapter *ancestor nodes* exist.
    2. Add a regulation node with full metadata.
    3. Add ``CONTAINS`` edges along the hierarchy.
    4. Add ``REFERENCES`` edges for each entry in ``cross_refs``.
    5. Add a ``DEFINES`` edge for ``.01`` definition regulations.

    Args:
        regulations: List of regulation dicts from
            :func:`~ingestion.xml_parser.parse_comar_xml`.
        save_path: Where to pickle the graph.  Pass ``None`` to skip saving.

    Returns:
        The constructed :class:`networkx.DiGraph`.
    """
    graph = nx.DiGraph()

    # Index all regulation chunk_ids → citation for resolving cross-refs
    citation_to_chunk_id: dict[str, str] = {
        reg["citation"]: reg["chunk_id"] for reg in regulations
    }

    ref_misses = 0

    for reg in regulations:
        t = reg["title_num"]
        sub = reg["subtitle_num"]
        ch = reg["chapter_num"]
        reg_id = reg["chunk_id"]

        tid = _title_id(t)
        sid = _subtitle_id(t, sub)
        cid = _chapter_id(t, sub, ch)

        # ── Ensure ancestor nodes ─────────────────────────────────────────
        if not graph.has_node(tid):
            graph.add_node(
                tid,
                node_type="title",
                title_num=t,
                title_name=reg.get("title_name", ""),
            )

        if not graph.has_node(sid):
            graph.add_node(
                sid,
                node_type="subtitle",
                title_num=t,
                subtitle_num=sub,
                subtitle_name=reg.get("subtitle_name", ""),
            )
            graph.add_edge(tid, sid, edge_type="CONTAINS")

        if not graph.has_node(cid):
            graph.add_node(
                cid,
                node_type="chapter",
                title_num=t,
                subtitle_num=sub,
                chapter_num=ch,
                chapter_name=reg.get("chapter_name", ""),
            )
            graph.add_edge(sid, cid, edge_type="CONTAINS")

        # ── Regulation node ───────────────────────────────────────────────
        graph.add_node(
            reg_id,
            node_type="regulation",
            **{k: v for k, v in reg.items() if k != "text"},
            # Omit full text from graph to keep pickle size small
        )
        graph.add_edge(cid, reg_id, edge_type="CONTAINS")

        # ── DEFINES edge for .01 definition regulations ───────────────────
        if reg.get("chunk_type") == "definition":
            graph.add_edge(reg_id, cid, edge_type="DEFINES")

    # ── REFERENCES edges (second pass — all nodes guaranteed to exist) ────
    for reg in regulations:
        src_id = reg["chunk_id"]
        for cited_citation in reg.get("cross_refs", []):
            tgt_id = citation_to_chunk_id.get(cited_citation)
            if tgt_id and tgt_id != src_id:
                graph.add_edge(src_id, tgt_id, edge_type="REFERENCES")
            elif tgt_id is None:
                ref_misses += 1

    # ── Summary ───────────────────────────────────────────────────────────
    contains_edges = sum(
        1 for _, _, d in graph.edges(data=True) if d.get("edge_type") == "CONTAINS"
    )
    references_edges = sum(
        1 for _, _, d in graph.edges(data=True) if d.get("edge_type") == "REFERENCES"
    )
    defines_edges = sum(
        1 for _, _, d in graph.edges(data=True) if d.get("edge_type") == "DEFINES"
    )

    logger.info(
        "Graph built: %d nodes  |  %d CONTAINS  |  %d REFERENCES  |  %d DEFINES",
        graph.number_of_nodes(),
        contains_edges,
        references_edges,
        defines_edges,
    )
    if ref_misses:
        logger.debug(
            "%d cross-reference targets not resolved (may cite other titles)", ref_misses
        )

    # ── Persist ───────────────────────────────────────────────────────────
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as fh:
            pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Graph saved → %s  (%.1f KB)", save_path, save_path.stat().st_size / 1024)

    return graph


# ── Query helpers ──────────────────────────────────────────────────────────────


def load_graph(path: Path = DEFAULT_GRAPH_PATH) -> nx.DiGraph:
    """Load a pickled graph from disk.

    Args:
        path: Path to the ``.pkl`` file.

    Returns:
        The loaded :class:`networkx.DiGraph`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    with open(path, "rb") as fh:
        graph: nx.DiGraph = pickle.load(fh)
    logger.info("Graph loaded from %s (%d nodes)", path, graph.number_of_nodes())
    return graph


def get_cross_refs(chunk_id: str, graph: nx.DiGraph) -> list[str]:
    """Return the chunk_ids of all regulations that *chunk_id* references.

    Follows ``REFERENCES`` edges from *chunk_id* outward.

    Args:
        chunk_id: Source regulation node id, e.g. ``"COMAR.15.01.01.03"``.
        graph: The knowledge graph.

    Returns:
        List of target chunk_id strings.  Empty list if node not found or no
        outgoing REFERENCES edges.
    """
    if not graph.has_node(chunk_id):
        logger.debug("get_cross_refs: node %s not in graph", chunk_id)
        return []
    return [
        tgt
        for _, tgt, data in graph.out_edges(chunk_id, data=True)
        if data.get("edge_type") == "REFERENCES"
    ]


def get_definitions_for_chapter(
    chapter_id: str, graph: nx.DiGraph
) -> list[str]:
    """Return chunk_ids of all definition regulations in a chapter.

    Finds regulation nodes with a ``DEFINES`` edge pointing TO *chapter_id*.

    Args:
        chapter_id: Chapter node id, e.g. ``"CHAPTER.15.01.01"``.
        graph: The knowledge graph.

    Returns:
        List of definition regulation chunk_id strings.
    """
    if not graph.has_node(chapter_id):
        logger.debug("get_definitions_for_chapter: node %s not in graph", chapter_id)
        return []
    return [
        src
        for src, _, data in graph.in_edges(chapter_id, data=True)
        if data.get("edge_type") == "DEFINES"
    ]


def get_chapter_for_regulation(
    chunk_id: str, graph: nx.DiGraph
) -> str | None:
    """Return the chapter node id for a given regulation chunk_id."""
    if not graph.has_node(chunk_id):
        return None
    for src, _, data in graph.in_edges(chunk_id, data=True):
        if data.get("edge_type") == "CONTAINS":
            node_data = graph.nodes[src]
            if node_data.get("node_type") == "chapter":
                return src
    return None
