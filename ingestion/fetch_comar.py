"""fetch_comar.py — Download COMAR XML files from GitHub.

Downloads Title XML files from the maryland-dsd/law-xml repository
(https://github.com/maryland-dsd/law-xml), which publishes the Code of
Maryland Regulations (COMAR) as structured XML under a CC0/CC-BY-SA license.

NOTE: The source repo is `maryland-dsd/law-xml`. Content transitions to CC0
public domain 180 days after publication (the project is colloquially called
"law-xml-cc0" in reference to that policy).

Directory layout inside DATA_DIR mirrors the repo::

    DATA_DIR/
    └── us/md/exec/comar/
        ├── 15/
        │   ├── index.xml
        │   ├── 01/
        │   │   ├── index.xml
        │   │   ├── 01.xml
        │   │   └── ...
        │   └── ...
        └── 26/
            └── ...

Usage::

    from ingestion.fetch_comar import fetch_comar_xml

    paths = fetch_comar_xml(["15", "26"], refresh=False)
    # {"15": PosixPath("./data/xml_cache/us/md/exec/comar/15/index.xml"), ...}

CLI::

    python ingestion/fetch_comar.py --titles 15 26 --refresh
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

GITHUB_API = "https://api.github.com"
REPO = "maryland-dsd/law-xml"
BRANCH = "main"
COMAR_BASE = "us/md/exec/comar"

# GitHub's unauthenticated rate limit is 60 req/hr; authenticated is 5000/hr.
# Set GITHUB_TOKEN in .env to raise the limit.
_GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN")
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        **({"Authorization": f"Bearer {_GITHUB_TOKEN}"} if _GITHUB_TOKEN else {}),
    }
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_data_dir() -> Path:
    """Return the XML cache directory, creating it if necessary."""
    data_dir = Path(os.getenv("DATA_DIR", "./data/xml_cache"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _github_get(url: str, params: dict[str, Any] | None = None) -> Any:
    """GET a GitHub API URL with simple retry-on-rate-limit logic.

    Args:
        url: Full API URL.
        params: Optional query parameters.

    Returns:
        Parsed JSON response.

    Raises:
        requests.HTTPError: On non-recoverable HTTP errors.
    """
    for attempt in range(3):
        resp = _SESSION.get(url, params=params, timeout=30)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset_ts - int(time.time()), 5)
            logger.warning("GitHub rate limit hit — waiting %ds (attempt %d/3)", wait, attempt + 1)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"GitHub API request failed after 3 retries: {url}")


def _download_raw(path_in_repo: str, dest: Path, refresh: bool) -> None:
    """Download a single raw file from GitHub and write it to *dest*.

    Args:
        path_in_repo: Repo-relative path, e.g. ``us/md/exec/comar/15/index.xml``.
        dest: Local destination path.
        refresh: If *False* and *dest* already exists, skip the download.
    """
    if dest.exists() and not refresh:
        logger.debug("Cache hit — skipping %s", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path_in_repo}"
    resp = _SESSION.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.debug("Downloaded %s → %s (%d bytes)", path_in_repo, dest, len(resp.content))


def _list_title_blobs(title: str) -> list[str]:
    """Return all repo-relative XML file paths under a given COMAR title.

    Uses the Git Trees API with ``recursive=1`` to get every blob under
    ``us/md/exec/comar/{title}/`` in a single request.

    Args:
        title: Zero-padded title number, e.g. ``"15"`` or ``"26"``.

    Returns:
        List of repo-relative file paths (strings) ending in ``.xml``.
    """
    tree_url = f"{GITHUB_API}/repos/{REPO}/git/trees/{BRANCH}"
    data = _github_get(tree_url, params={"recursive": "1"})

    if data.get("truncated"):
        logger.warning(
            "GitHub tree response was truncated — some files may be missing. "
            "Consider setting GITHUB_TOKEN to increase rate limits."
        )

    prefix = f"{COMAR_BASE}/{title}/"
    blobs = [
        item["path"]
        for item in data.get("tree", [])
        if item["type"] == "blob"
        and item["path"].startswith(prefix)
        and item["path"].endswith(".xml")
    ]
    logger.info("Found %d XML blobs for Title %s", len(blobs), title)
    return blobs


# ── Public API ────────────────────────────────────────────────────────────────


def fetch_comar_xml(
    titles: list[str] = ["15", "26"],  # noqa: B006
    refresh: bool = False,
) -> dict[str, Path]:
    """Download COMAR XML files for the requested titles.

    Files are cached on disk under *DATA_DIR* (from the environment).  Pass
    ``refresh=True`` to force re-download even when cached files exist.

    Args:
        titles: List of COMAR title numbers to download, e.g. ``["15", "26"]``.
                Numbers are zero-padded to two digits automatically.
        refresh: When ``True``, re-download files even if they are cached.

    Returns:
        Mapping from title number string (e.g. ``"15"``) to the local path of
        the title's ``index.xml`` file.

    Example::

        paths = fetch_comar_xml(["15", "26"])
        # {"15": Path("./data/xml_cache/us/md/exec/comar/15/index.xml"),
        #  "26": Path("./data/xml_cache/us/md/exec/comar/26/index.xml")}
    """
    data_dir = _get_data_dir()
    # Normalise title numbers to zero-padded strings (e.g. "5" → "05")
    normalised = [t.zfill(2) for t in titles]
    result: dict[str, Path] = {}

    for title in normalised:
        logger.info("── Fetching Title %s ─────────────────────────────────", title)

        blobs = _list_title_blobs(title)
        if not blobs:
            logger.error("No XML blobs found for Title %s — check the repo layout.", title)
            continue

        with tqdm(blobs, desc=f"Title {title}", unit="file", leave=True) as pbar:
            for repo_path in pbar:
                dest = data_dir / repo_path
                pbar.set_postfix_str(Path(repo_path).name, refresh=False)
                _download_raw(repo_path, dest, refresh)

        index_path = data_dir / COMAR_BASE / title / "index.xml"
        if not index_path.exists():
            logger.error("index.xml not found for Title %s at %s", title, index_path)
        else:
            result[title] = index_path
            logger.info("Title %s ready → %s", title, index_path)

    return result


# ── CLI entry point ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download COMAR XML files from GitHub for the specified titles."
    )
    p.add_argument(
        "--titles",
        nargs="+",
        default=["15", "26"],
        metavar="N",
        help="COMAR title numbers to download (default: 15 26)",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download even when cached files exist",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    paths = fetch_comar_xml(titles=args.titles, refresh=args.refresh)

    print("\nDownloaded title index paths:")
    for title, path in sorted(paths.items()):
        print(f"  Title {title} → {path}")
