"""run_ingestion.py — Master orchestration script for COMAR RAG ingestion.

Runs the full data pipeline in order:

1.  Fetch XML files from GitHub (Title 15 & 26) — cached if already present
2.  Parse each title into a list of regulation dicts
3.  Chunk regulations → (chunk list, definitions lookup)
4.  Build a NetworkX knowledge graph and save to ./data/comar_graph.pkl
5.  Initialise the BGE-M3 embedder
6.  Upload all chunks (dense + sparse vectors) to Qdrant
7.  Save definitions lookup to ./data/definitions.json
8.  Print a final summary report

CLI::

    python ingestion/run_ingestion.py
    python ingestion/run_ingestion.py --refresh       # force re-download XML
    python ingestion/run_ingestion.py --skip-fetch    # assume XML already cached
    python ingestion/run_ingestion.py --titles 15     # process only one title
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Logging config (set before any local imports) ─────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("run_ingestion")

# ── Local imports ─────────────────────────────────────────────────────────────

from ingestion.chunker import create_chunks
from ingestion.embedder import Embedder
from ingestion.fetch_comar import fetch_comar_xml
from ingestion.graph_builder import build_knowledge_graph
from ingestion.qdrant_uploader import upload_chunks, verify_collection
from ingestion.xml_parser import parse_comar_xml

# ── Constants ─────────────────────────────────────────────────────────────────

DEFINITIONS_PATH = Path("./data/definitions.json")
GRAPH_PATH = Path("./data/comar_graph.pkl")
DEFAULT_TITLES = ["15", "26"]


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the full COMAR RAG ingestion pipeline."
    )
    p.add_argument(
        "--titles",
        nargs="+",
        default=DEFAULT_TITLES,
        metavar="N",
        help="COMAR title numbers to ingest (default: 15 26)",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of XML files even when cached",
    )
    p.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip GitHub download — assume XML already cached in DATA_DIR",
    )
    p.add_argument(
        "--skip-upload",
        action="store_true",
        help="Build chunks and graph but do NOT upload to Qdrant",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


# ── Pipeline steps ─────────────────────────────────────────────────────────────


def _step_fetch(titles: list[str], refresh: bool) -> dict[str, Path]:
    logger.info("━━━ Step 1/7: Fetch COMAR XML ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    paths = fetch_comar_xml(titles=titles, refresh=refresh)
    if not paths:
        logger.error("No XML paths returned — aborting.")
        sys.exit(1)
    return paths


def _step_fetch_from_cache(titles: list[str]) -> dict[str, Path]:
    """Resolve title index.xml paths from the local DATA_DIR cache."""
    import os
    data_dir = Path(os.getenv("DATA_DIR", "./data/xml_cache"))
    paths: dict[str, Path] = {}
    for t in titles:
        t_padded = t.zfill(2)
        p = data_dir / "us/md/exec/comar" / t_padded / "index.xml"
        if not p.exists():
            logger.error(
                "Cached index.xml not found for Title %s at %s. "
                "Run without --skip-fetch first.",
                t_padded,
                p,
            )
            sys.exit(1)
        paths[t_padded] = p
    return paths


def _step_parse(paths: dict[str, Path]) -> list[dict]:
    logger.info("━━━ Step 2/7: Parse XML ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    all_regs: list[dict] = []
    for title, path in sorted(paths.items()):
        regs = parse_comar_xml(path)
        logger.info("  Title %s → %d regulations", title, len(regs))
        all_regs.extend(regs)
    logger.info("Total regulations: %d", len(all_regs))
    return all_regs


def _step_chunk(regulations: list[dict]) -> tuple[list[dict], dict]:
    logger.info("━━━ Step 3/7: Create Chunks ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    chunks, definitions = create_chunks(regulations)

    primary = sum(1 for c in chunks if c.get("chunk_type") in ("regulation", "definition"))
    subsections = sum(1 for c in chunks if c.get("chunk_type") == "subsection")
    logger.info(
        "  Primary chunks: %d  |  Subsections: %d  |  Total: %d",
        primary,
        subsections,
        len(chunks),
    )
    logger.info("  Definitions extracted: %d", len(definitions))
    return chunks, definitions


def _step_build_graph(regulations: list[dict]) -> None:
    logger.info("━━━ Step 4/7: Build Knowledge Graph ━━━━━━━━━━━━━━━━━━━━━━━━")
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph = build_knowledge_graph(regulations, save_path=GRAPH_PATH)
    logger.info(
        "  Graph: %d nodes, %d edges → %s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        GRAPH_PATH,
    )


def _step_init_embedder() -> Embedder:
    logger.info("━━━ Step 5/7: Initialise Embedder ━━━━━━━━━━━━━━━━━━━━━━━━━")
    emb = Embedder()
    emb._load()  # load now so timing is visible
    return emb


def _step_upload(chunks: list[dict], embedder: Embedder) -> dict:
    logger.info("━━━ Step 6/7: Upload to Qdrant ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    upload_chunks(chunks, embedder)
    stats = verify_collection()
    logger.info(
        "  Qdrant '%s': %s points uploaded, status=%s",
        stats["name"],
        stats["points_count"],
        stats["status"],
    )
    return stats


def _step_save_definitions(definitions: dict) -> None:
    logger.info("━━━ Step 7/7: Save Definitions Lookup ━━━━━━━━━━━━━━━━━━━━━")
    DEFINITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFINITIONS_PATH, "w", encoding="utf-8") as fh:
        json.dump(definitions, fh, indent=2, ensure_ascii=False)
    size_kb = DEFINITIONS_PATH.stat().st_size / 1024
    logger.info(
        "  Saved %d definitions → %s (%.1f KB)",
        len(definitions),
        DEFINITIONS_PATH,
        size_kb,
    )


# ── Summary printer ────────────────────────────────────────────────────────────


def _print_summary(
    regulations: list[dict],
    chunks: list[dict],
    definitions: dict,
    qdrant_stats: dict | None,
    elapsed: float,
) -> None:
    per_title: dict[str, int] = {}
    for r in regulations:
        t = r["title_num"]
        per_title[t] = per_title.get(t, 0) + 1

    primary = sum(1 for c in chunks if c.get("chunk_type") in ("regulation", "definition"))
    subsections = sum(1 for c in chunks if c.get("chunk_type") == "subsection")
    avg_tokens = (
        sum(c.get("token_count", 0) for c in chunks) / len(chunks) if chunks else 0
    )

    print()
    print("=" * 65)
    print("  COMAR RAG — Ingestion Complete")
    print("=" * 65)
    print(f"  Elapsed time          : {elapsed:.1f}s")
    print()
    print("  REGULATIONS")
    for t, count in sorted(per_title.items()):
        print(f"    Title {t}            : {count:>5}")
    print(f"    Total               : {len(regulations):>5}")
    print()
    print("  CHUNKS")
    print(f"    Primary             : {primary:>5}")
    print(f"    Subsections         : {subsections:>5}")
    print(f"    Total               : {len(chunks):>5}")
    print(f"    Avg tokens / chunk  : {avg_tokens:>7.1f}")
    print()
    print("  DEFINITIONS LOOKUP")
    print(f"    Terms extracted     : {len(definitions):>5}")
    print(f"    Saved to            : {DEFINITIONS_PATH}")
    print()
    print("  GRAPH")
    print(f"    Saved to            : {GRAPH_PATH}")
    print()
    if qdrant_stats:
        print("  QDRANT")
        print(f"    Collection          : {qdrant_stats['name']}")
        print(f"    Vectors uploaded    : {qdrant_stats['points_count']:>5}")
        print(f"    Status              : {qdrant_stats['status']}")
    else:
        print("  QDRANT  : upload skipped (--skip-upload)")
    print("=" * 65)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _build_parser().parse_args()

    # Update log level from CLI arg
    logging.getLogger().setLevel(args.log_level)

    start = time.monotonic()

    # Step 1 — Fetch / resolve XML paths
    if args.skip_fetch:
        logger.info("━━━ Step 1/7: Fetch COMAR XML (SKIPPED — using cache) ━━━")
        xml_paths = _step_fetch_from_cache(args.titles)
    else:
        xml_paths = _step_fetch(args.titles, args.refresh)

    # Step 2 — Parse
    regulations = _step_parse(xml_paths)

    # Step 3 — Chunk
    chunks, definitions = _step_chunk(regulations)

    # Step 4 — Build graph
    _step_build_graph(regulations)

    # Steps 5 & 6 — Embed + Upload (can be skipped for offline testing)
    qdrant_stats: dict | None = None
    if not args.skip_upload:
        embedder = _step_init_embedder()
        qdrant_stats = _step_upload(chunks, embedder)
    else:
        logger.info("━━━ Steps 5–6/7: Embed + Upload (SKIPPED) ━━━━━━━━━━━━━━")

    # Step 7 — Persist definitions
    _step_save_definitions(definitions)

    elapsed = time.monotonic() - start
    _print_summary(regulations, chunks, definitions, qdrant_stats, elapsed)


if __name__ == "__main__":
    main()
