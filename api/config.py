"""api/config.py — Settings loaded from .env."""
from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


@lru_cache(maxsize=1)
def _qdrant_health_client(host: str, port: int):
    """Lightweight singleton QdrantClient for health checks (avoids circular import)."""
    from qdrant_client import QdrantClient
    return QdrantClient(host=host, port=port, timeout=5)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM — DeepSeek (primary)
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    # LLM — fallbacks
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "comar_regulations"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "comar_password"

    # Embeddings
    bge_m3_device: str = "cpu"

    # Data
    data_dir: str = "./data/xml_cache"

    @property
    def qdrant_ready(self) -> bool:
        """True when Qdrant is reachable and collection is populated."""
        try:
            from qdrant_client import QdrantClient
            c = _qdrant_health_client(self.qdrant_host, self.qdrant_port)
            info = c.get_collection(self.qdrant_collection)
            return (info.points_count or 0) > 0
        except Exception:
            return False

    @property
    def llm_ready(self) -> bool:
        """True when any configured LLM API key is valid."""
        ph = "your_key_here"
        if self.deepseek_api_key and self.deepseek_api_key != ph:
            return True
        if self.anthropic_api_key and self.anthropic_api_key != ph:
            return True
        if self.openai_api_key and self.openai_api_key != ph:
            return True
        return False


@lru_cache
def get_settings() -> Settings:
    return Settings()
