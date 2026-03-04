"""xml_parser.py — Parse COMAR XML into structured regulation chunks.

Takes a COMAR title ``index.xml`` (as returned by :func:`fetch_comar_xml`),
resolves all ``xi:include`` references, and walks the document hierarchy::

    Title ▶ Subtitle ▶ Chapter ▶ Section (Regulation)

Each regulation becomes one dict suitable for embedding and storage.

Schema of returned dicts::

    {
        "chunk_id":       str,   # "COMAR.15.05.01.06"
        "citation":       str,   # "COMAR 15.05.01.06"
        "title_num":      str,   # "15"
        "subtitle_num":   str,   # "05"
        "chapter_num":    str,   # "01"
        "regulation_num": str,   # "06"
        "title_name":     str,
        "subtitle_name":  str,
        "chapter_name":   str,
        "regulation_name":str,
        "text":           str,   # full plain text, whitespace-normalised
        "effective_date": str | None,  # "YYYY-MM-DD"
        "cross_refs":     list[str],   # ["COMAR 15.01.01.03", ...]
        "chunk_type":     str,   # "definition" | "regulation"
    }

Usage::

    from ingestion.xml_parser import parse_comar_xml
    regs = parse_comar_xml(Path("data/xml_cache/us/md/exec/comar/15/index.xml"))

CLI::

    python ingestion/xml_parser.py --path data/xml_cache/us/md/exec/comar/15/index.xml
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

from lxml import etree

logger = logging.getLogger(__name__)

# ── XML namespaces ────────────────────────────────────────────────────────────

NS_LIB = "https://open.law/schemas/library"
NS_XI = "http://www.w3.org/2001/XInclude"
NS_CACHE = "https://open.law/schemas/cache"

_TAG_CONTAINER = f"{{{NS_LIB}}}container"
_TAG_SECTION = f"{{{NS_LIB}}}section"
_TAG_PREFIX = f"{{{NS_LIB}}}prefix"
_TAG_NUM = f"{{{NS_LIB}}}num"
_TAG_HEADING = f"{{{NS_LIB}}}heading"
_TAG_PARA = f"{{{NS_LIB}}}para"
_TAG_TEXT = f"{{{NS_LIB}}}text"
_TAG_ANNOTATION = f"{{{NS_LIB}}}annotation"
_TAG_ANNOTATIONS = f"{{{NS_LIB}}}annotations"
_TAG_CITE = f"{{{NS_LIB}}}cite"
_TAG_XI_INCLUDE = f"{{{NS_XI}}}include"

# Fallback: some elements may appear without a namespace prefix
_BARE = {
    "container": "container",
    "section": "section",
    "prefix": "prefix",
    "num": "num",
    "heading": "heading",
    "text": "text",
    "annotation": "annotation",
    "cite": "cite",
}

# Pattern matching COMAR pipe-delimited path attributes
# e.g. "15|01|01|.03" or "|15|01|01|.03|C."
_COMAR_PATH_RE = re.compile(r"(\d+)\|(\d+)\|(\d+)\|(\.\d+)")


# ── xi:include resolver ───────────────────────────────────────────────────────


def _resolve_xincludes(element: etree._Element, base_dir: Path, depth: int = 0) -> None:
    """Recursively replace ``xi:include`` elements with the loaded file content.

    The function modifies *element* in-place.  Included documents are loaded
    relative to *base_dir* and themselves get their xi:includes resolved.

    Args:
        element: The lxml element whose xi:include children should be resolved.
        base_dir: Directory to resolve relative href values against.
        depth: Recursion depth guard (aborts at 20 to prevent infinite loops).
    """
    if depth > 20:
        logger.warning("xi:include recursion depth limit reached — aborting branch")
        return

    for xi in element.findall(f".//{_TAG_XI_INCLUDE}"):
        href = xi.get("href")
        if not href:
            continue

        include_path = (base_dir / href).resolve()
        if not include_path.exists():
            logger.warning("xi:include target not found: %s", include_path)
            # Replace with an empty placeholder so iteration doesn't break
            parent = xi.getparent()
            if parent is not None:
                parent.remove(xi)
            continue

        try:
            included_tree = etree.parse(str(include_path))
            included_root = included_tree.getroot()
        except etree.XMLSyntaxError as exc:
            logger.error("Failed to parse %s: %s", include_path, exc)
            parent = xi.getparent()
            if parent is not None:
                parent.remove(xi)
            continue

        # Resolve xi:includes in the included document first
        _resolve_xincludes(included_root, include_path.parent, depth + 1)

        parent = xi.getparent()
        if parent is None:
            continue

        idx = list(parent).index(xi)

        xpointer = xi.get("xpointer", "")
        if xpointer:
            # xpointer="xpointer(/container/*)" — insert children of root
            children = list(included_root)
        else:
            # No xpointer — insert the whole root element
            children = [included_root]

        parent.remove(xi)
        for offset, child in enumerate(children):
            parent.insert(idx + offset, child)

    logger.debug("xi:include resolution complete at depth %d for %s", depth, base_dir)


# ── Text extraction helpers ───────────────────────────────────────────────────


def _get_child_text(element: etree._Element, local_name: str) -> str:
    """Return stripped text of the first direct child whose local-name matches.

    Handles both namespaced (``{ns}tag``) and bare (``tag``) elements.
    """
    for child in element:
        tag = etree.QName(child.tag).localname if child.tag is not None else ""
        if tag == local_name:
            return (child.text or "").strip()
    return ""


def _element_text(element: etree._Element) -> str:
    """Recursively collect all text content from an element tree.

    Skips ``<annotations>`` blocks (administrative history, authority).
    Returns whitespace-normalised text.
    """
    parts: list[str] = []

    def _walk(el: etree._Element) -> None:
        local = etree.QName(el.tag).localname if el.tag is not None else ""
        # Skip annotations — they contain history, not regulatory text
        if local in ("annotations", "annotation"):
            return
        if el.text:
            parts.append(el.text.strip())
        for child in el:
            _walk(child)
            if child.tail:
                parts.append(child.tail.strip())

    _walk(element)
    text = " ".join(p for p in parts if p)
    # Collapse runs of whitespace
    return re.sub(r"\s{2,}", " ", text).strip()


def _extract_cross_refs(section: etree._Element) -> list[str]:
    """Extract COMAR cross-references from ``<cite>`` path attributes.

    Looks for ``path`` attributes matching the pattern ``15|01|01|.03``.

    Args:
        section: The ``<section>`` element to inspect.

    Returns:
        Deduplicated list of COMAR citation strings, e.g. ``["COMAR 15.01.01.03"]``.
    """
    refs: list[str] = []
    for cite in section.iter(_TAG_CITE):
        path = cite.get("path", "")
        m = _COMAR_PATH_RE.search(path)
        if m:
            t, sub, ch, reg = m.groups()
            reg_clean = reg.lstrip(".")
            refs.append(f"COMAR {t}.{sub}.{ch}.{reg_clean}")
    # Also check href attributes on any element (some schemas use those)
    for el in section.iter():
        href = el.get("href") or el.get(f"{{{NS_CACHE}}}ref-path", "")
        m = _COMAR_PATH_RE.search(href)
        if m:
            t, sub, ch, reg = m.groups()
            reg_clean = reg.lstrip(".")
            refs.append(f"COMAR {t}.{sub}.{ch}.{reg_clean}")
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def _extract_effective_date(section: etree._Element) -> str | None:
    """Return the most recent effective date from a section's annotations.

    Searches ``<annotation type="History">`` elements for the ``effective``
    attribute and returns the lexicographically largest value (latest date).

    Args:
        section: The ``<section>`` element.

    Returns:
        ISO date string ``"YYYY-MM-DD"`` or ``None`` if not found.
    """
    dates: list[str] = []
    for ann in section.iter(_TAG_ANNOTATION):
        eff = ann.get("effective")
        if eff:
            dates.append(eff)
    return max(dates) if dates else None


# ── Hierarchy walkers ─────────────────────────────────────────────────────────


def _local(element: etree._Element) -> str:
    """Return the local-name of an lxml element (strips namespace prefix)."""
    return etree.QName(element.tag).localname if element.tag is not None else ""


def _container_prefix(container: etree._Element) -> str:
    """Return the ``<prefix>`` text of a container element, lowercased."""
    return _get_child_text(container, "prefix").lower()


def _container_num(container: etree._Element) -> str:
    """Return the ``<num>`` text of a container element, stripped."""
    return _get_child_text(container, "num").strip()


def _container_heading(container: etree._Element) -> str:
    """Return the ``<heading>`` text of a container element."""
    return _get_child_text(container, "heading")


def _parse_regulation_num(raw: str) -> str:
    """Strip the leading dot from a regulation ``<num>`` value.

    Examples:
        ``".01"`` → ``"01"``
        ``"01"`` → ``"01"`` (unchanged)
    """
    return raw.lstrip(".").strip()


def _iter_containers_by_prefix(
    parent: etree._Element, prefix_value: str
) -> list[etree._Element]:
    """Return direct-child ``<container>`` elements with a given prefix value.

    Args:
        parent: Element to search within.
        prefix_value: Lowercase prefix to match, e.g. ``"subtitle"``.

    Returns:
        List of matching container elements.
    """
    results = []
    for child in parent:
        if _local(child) == "container" and _container_prefix(child) == prefix_value:
            results.append(child)
    return results


def _iter_sections(parent: etree._Element) -> list[etree._Element]:
    """Return all direct-child ``<section>`` elements of *parent*."""
    return [child for child in parent if _local(child) == "section"]


# ── Core parser ───────────────────────────────────────────────────────────────


def _parse_chapter(
    chapter_el: etree._Element,
    title_num: str,
    title_name: str,
    subtitle_num: str,
    subtitle_name: str,
) -> list[dict[str, Any]]:
    """Extract all regulations from a single ``<container prefix="Chapter">``.

    Args:
        chapter_el: The chapter container element.
        title_num: Parent title number string (e.g. ``"15"``).
        title_name: Parent title heading.
        subtitle_num: Parent subtitle number string (e.g. ``"01"``).
        subtitle_name: Parent subtitle heading.

    Returns:
        List of regulation dicts (one per ``<section>``).
    """
    chapter_num = _container_num(chapter_el).zfill(2)
    chapter_name = _container_heading(chapter_el)

    regulations: list[dict[str, Any]] = []

    for section in _iter_sections(chapter_el):
        raw_num = _get_child_text(section, "num")
        reg_num = _parse_regulation_num(raw_num).zfill(2)
        reg_name = _get_child_text(section, "heading")

        # Build canonical identifiers
        chunk_id = f"COMAR.{title_num}.{subtitle_num}.{chapter_num}.{reg_num}"
        citation = f"COMAR {title_num}.{subtitle_num}.{chapter_num}.{reg_num}"

        # Determine chunk type — regulations numbered ".01" are definitions
        chunk_type = "definition" if raw_num.lstrip(".") == "01" else "regulation"

        text = _element_text(section)
        effective_date = _extract_effective_date(section)
        cross_refs = _extract_cross_refs(section)

        regulations.append(
            {
                "chunk_id": chunk_id,
                "citation": citation,
                "title_num": title_num,
                "subtitle_num": subtitle_num,
                "chapter_num": chapter_num,
                "regulation_num": reg_num,
                "title_name": title_name,
                "subtitle_name": subtitle_name,
                "chapter_name": chapter_name,
                "regulation_name": reg_name,
                "text": text,
                "effective_date": effective_date,
                "cross_refs": cross_refs,
                "chunk_type": chunk_type,
            }
        )

    return regulations


def parse_comar_xml(xml_path: Path) -> list[dict[str, Any]]:
    """Parse a COMAR XML file and return a list of regulation dicts.

    Accepts either:
    - A title ``index.xml`` file (will resolve all ``xi:include`` references)
    - A single chapter ``.xml`` file (will attempt to infer title/subtitle
      numbers from the file path if they cannot be found in the document)

    The function handles the full ``xi:include`` resolution so that the caller
    does not need to pre-process the file tree.

    Args:
        xml_path: Absolute or relative :class:`~pathlib.Path` to the XML file.

    Returns:
        List of regulation dicts.  See module docstring for field descriptions.

    Raises:
        FileNotFoundError: If *xml_path* does not exist.
        etree.XMLSyntaxError: If the file is not valid XML.
    """
    xml_path = Path(xml_path).resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    logger.info("Parsing %s", xml_path)
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    # Resolve xi:includes relative to the directory containing the XML file
    _resolve_xincludes(root, xml_path.parent)

    # ── Navigate the hierarchy ─────────────────────────────────────────────

    all_regulations: list[dict[str, Any]] = []

    root_prefix = _container_prefix(root)

    if root_prefix == "title":
        # Top-level call: Title → Subtitle → Chapter → Section
        title_num = _container_num(root).zfill(2)
        title_name = _container_heading(root)

        subtitles = _iter_containers_by_prefix(root, "subtitle")
        if not subtitles:
            logger.warning("No subtitle containers found in %s", xml_path)

        for subtitle_el in subtitles:
            subtitle_num = _container_num(subtitle_el).zfill(2)
            subtitle_name = _container_heading(subtitle_el)

            chapters = _iter_containers_by_prefix(subtitle_el, "chapter")
            for chapter_el in chapters:
                regs = _parse_chapter(
                    chapter_el,
                    title_num=title_num,
                    title_name=title_name,
                    subtitle_num=subtitle_num,
                    subtitle_name=subtitle_name,
                )
                all_regulations.extend(regs)

    elif root_prefix == "subtitle":
        # Subtitle-level file — infer title from path
        title_num = _infer_from_path(xml_path, level="title")
        title_name = ""
        subtitle_num = _container_num(root).zfill(2)
        subtitle_name = _container_heading(root)

        for chapter_el in _iter_containers_by_prefix(root, "chapter"):
            regs = _parse_chapter(
                chapter_el,
                title_num=title_num,
                title_name=title_name,
                subtitle_num=subtitle_num,
                subtitle_name=subtitle_name,
            )
            all_regulations.extend(regs)

    elif root_prefix == "chapter":
        # Chapter-level file — infer title/subtitle from path
        title_num = _infer_from_path(xml_path, level="title")
        subtitle_num = _infer_from_path(xml_path, level="subtitle")
        title_name = ""
        subtitle_name = ""
        regs = _parse_chapter(
            root,
            title_num=title_num,
            title_name=title_name,
            subtitle_num=subtitle_num,
            subtitle_name=subtitle_name,
        )
        all_regulations.extend(regs)

    else:
        logger.warning(
            "Unexpected root container prefix %r in %s — attempting chapter parse",
            root_prefix,
            xml_path,
        )
        # Best-effort: treat root as a chapter
        title_num = _infer_from_path(xml_path, level="title")
        subtitle_num = _infer_from_path(xml_path, level="subtitle")
        regs = _parse_chapter(
            root,
            title_num=title_num,
            title_name="",
            subtitle_num=subtitle_num,
            subtitle_name="",
        )
        all_regulations.extend(regs)

    logger.info(
        "Parsed %d regulations from %s", len(all_regulations), xml_path.name
    )
    return all_regulations


# ── Path-based inference ──────────────────────────────────────────────────────


def _infer_from_path(xml_path: Path, level: str) -> str:
    """Infer a title or subtitle number from the file's directory path.

    Expects paths of the form ``.../{comar_base}/{title}/{subtitle}/{file}.xml``.

    Args:
        xml_path: Path to the XML file.
        level: ``"title"`` or ``"subtitle"``.

    Returns:
        Two-digit zero-padded number string, or ``"00"`` if not determinable.
    """
    parts = xml_path.parts
    # Find the index of "comar" in the path and use relative offsets
    try:
        comar_idx = next(i for i, p in enumerate(parts) if p == "comar")
        if level == "title":
            return parts[comar_idx + 1].zfill(2)
        if level == "subtitle":
            return parts[comar_idx + 2].zfill(2)
    except (StopIteration, IndexError):
        pass
    logger.debug("Could not infer %s from path %s", level, xml_path)
    return "00"


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI smoke test — parse a title XML and print a summary."""
    parser = argparse.ArgumentParser(
        description="Parse a COMAR XML file and print summary statistics."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help=(
            "Path to a COMAR XML file (title index.xml or chapter .xml). "
            "Defaults to both Title 15 and Title 26 if they exist in DATA_DIR."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    import os
    from dotenv import load_dotenv

    load_dotenv()
    data_dir = Path(os.getenv("DATA_DIR", "./data/xml_cache"))

    if args.path:
        targets = [args.path]
    else:
        targets = [
            data_dir / "us/md/exec/comar/15/index.xml",
            data_dir / "us/md/exec/comar/26/index.xml",
        ]

    for target in targets:
        if not target.exists():
            print(f"\n[SKIP] {target} — file not found (run fetch_comar.py first)")
            continue

        print(f"\n{'='*60}")
        print(f" Parsing: {target}")
        print("=" * 60)

        regs = parse_comar_xml(target)

        print(f"  Total regulations parsed : {len(regs)}")

        # Count by type
        defs = sum(1 for r in regs if r["chunk_type"] == "definition")
        print(f"  Definition sections      : {defs}")
        print(f"  Regulatory sections      : {len(regs) - defs}")

        # Unique chapter/subtitle counts
        subtitles = {r["subtitle_num"] for r in regs}
        chapters = {(r["subtitle_num"], r["chapter_num"]) for r in regs}
        print(f"  Subtitles                : {len(subtitles)}")
        print(f"  Chapters                 : {len(chapters)}")

        print("\n  First 3 regulations:")
        for i, reg in enumerate(regs[:3]):
            print(f"\n  [{i+1}] {reg['citation']}")
            print(f"       Title   : {reg['title_name']}")
            print(f"       Subtitle: {reg['subtitle_name']}")
            print(f"       Chapter : {reg['chapter_name']}")
            print(f"       Reg     : {reg['regulation_name']}")
            print(f"       Type    : {reg['chunk_type']}")
            print(f"       Eff.    : {reg['effective_date']}")
            print(f"       XRefs   : {reg['cross_refs'][:3]}")
            snippet = reg["text"][:200].replace("\n", " ")
            print(f"       Text    : {snippet}…")


if __name__ == "__main__":
    main()
