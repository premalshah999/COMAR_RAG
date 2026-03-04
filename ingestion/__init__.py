"""COMAR RAG — ingestion package.

Provides XML fetch and parse utilities for the Code of Maryland Regulations.
"""

from ingestion.fetch_comar import fetch_comar_xml
from ingestion.xml_parser import parse_comar_xml

__all__ = ["fetch_comar_xml", "parse_comar_xml"]
