"""
StockM v1.0 - Phase 10, Lesson 3
RAG Configuration
==================

Loads RAG settings from the environment + a YAML config, following the same
pattern as the rest of StockM (env vars first, YAML for structure, frozen
Pydantic model). One ``RAGSettings`` object is built once and injected into
every RAG module - no module reads the filesystem or env directly.

Env vars (all optional - sensible defaults; see .env.example):
    RAG_EMBEDDING_MODEL     sentence-transformers model id (Lesson 5)
    RAG_VECTOR_DB_DIR       persistence dir for ChromaDB (Lesson 6)
    RAG_COLLECTION_NAME     ChromaDB collection name
    RAG_CHUNK_SIZE          target chunk size in characters (Lesson 4)
    RAG_CHUNK_OVERLAP       overlap between adjacent chunks (Lesson 4)
    RAG_TOP_K               default retrieved chunks per query (Lesson 7)
    RAG_LLM_PROVIDER        openai | local | none (Lesson 8)
    RAG_LLM_MODEL           model id for the generator (Lesson 8)

The YAML file (configs/rag_config.yaml) holds source-specific loader config
(RSS URLs, filings dirs). Kept out of env vars because it's structured data.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger("stockm.rag.config")

# Match the project-root resolution convention used everywhere else in StockM
# (see src/collectors/historical_collector.py):
#   __file__ = .../src/stockM/rag/config.py
#   parents[0] = .../rag
#   parents[1] = .../stockM
#   parents[2] = .../src
#   parents[3] = .../stockM   <- project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAG_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag_config.yaml"


class RAGSettings(BaseModel):
    """Frozen, immutable RAG settings. Built once via :func:`get_rag_settings`.

    Every RAG module receives a reference to this object (dependency injection)
    rather than reading config itself - makes modules testable (inject a fake
    settings object) and reproducible (same settings => same behavior).
    """

    model_config = {"frozen": True}

    # --- Embeddings (Lesson 5) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Small, fast, 384-dim. Good default for English finance text; swappable
    # to a finance-tuned model via env var with no code change.

    # --- Vector store (Lesson 6) ---
    vector_db_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "vector_db")
    collection_name: str = "stockm_rag"
    # ChromaDB persists to disk here; survives restarts (Lesson 13).

    # --- Chunking (Lesson 4) ---
    chunk_size: int = 400          # characters; tuned for finance news paragraphs
    chunk_overlap: int = 50        # ~12% overlap; preserves cross-boundary context
    min_chunk_size: int = 50       # drop tiny trailing chunks with no signal

    # --- Retrieval (Lesson 7) ---
    top_k: int = 5                 # default context pool size for the generator

    # --- LLM (Lesson 8) ---
    llm_provider: str = "none"     # none = rule-based fallback generator (no API key needed)
    llm_model: str = "gpt-4o-mini" # only used when provider != none
    llm_api_key: str | None = None
    llm_base_url: str | None = None  # for local LLMs (Ollama, vLLM)

    # --- Documents ---
    documents_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "documents")

    @property
    def vector_db_dir_str(self) -> str:
        """ChromaDB's client wants a str path, not Path - normalize once."""
        return str(self.vector_db_dir)


def _load_yaml_config() -> dict[str, Any]:
    """Load configs/rag_config.yaml if present. Empty dict if missing.

    The YAML holds structured source config (RSS URLs, etc.) that doesn't fit
    env vars. Missing file is fine - we ship sensible defaults and a sample
    data loader that needs no external sources to run end-to-end.
    """
    if not RAG_CONFIG_PATH.exists():
        return {}
    try:
        with RAG_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            logger.warning("rag_config.yaml top-level must be a mapping; got %s", type(data))
            return {}
        return data
    except Exception as exc:  # noqa: BLE001 - config load must never crash the app
        logger.warning("Failed to load %s: %s", RAG_CONFIG_PATH, exc)
        return {}


def _env(name: str, default: str | None = None) -> str | None:
    """Read an env var. Real env wins; a project-root .env is a fallback.

    Mirrors the Phase 9 API config logic (src/api/config.py) - no python-dotenv
    dependency, just a tiny parser cached at import time.
    """
    val = os.environ.get(name)
    if val is not None:
        return val
    return _DOTENV.get(name, default)


_DOTENV: dict[str, str] = {}


def _load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        _DOTENV[key.strip()] = value.strip().strip('"').strip("'")


_load_dotenv()


def get_rag_settings() -> RAGSettings:
    """Build the RAGSettings from env + YAML. Call once, reuse the result.

    Resolution order: real env var > .env file > YAML > code default.
    Env wins so Docker/k8s overrides take precedence (same as Phase 9).
    """
    # Embeddings / store / chunking / retrieval / LLM come from env vars.
    kwargs: dict[str, Any] = {}

    if (v := _env("RAG_EMBEDDING_MODEL")) is not None:
        kwargs["embedding_model"] = v
    if (v := _env("RAG_VECTOR_DB_DIR")) is not None:
        kwargs["vector_db_dir"] = Path(v)
    if (v := _env("RAG_COLLECTION_NAME")) is not None:
        kwargs["collection_name"] = v
    if (v := _env("RAG_CHUNK_SIZE")) is not None:
        kwargs["chunk_size"] = int(v)
    if (v := _env("RAG_CHUNK_OVERLAP")) is not None:
        kwargs["chunk_overlap"] = int(v)
    if (v := _env("RAG_TOP_K")) is not None:
        kwargs["top_k"] = int(v)
    if (v := _env("RAG_LLM_PROVIDER")) is not None:
        kwargs["llm_provider"] = v
    if (v := _env("RAG_LLM_MODEL")) is not None:
        kwargs["llm_model"] = v
    if (v := _env("RAG_LLM_API_KEY")) is not None:
        kwargs["llm_api_key"] = v
    if (v := _env("RAG_LLM_BASE_URL")) is not None:
        kwargs["llm_base_url"] = v

    settings = RAGSettings(**kwargs)
    logger.debug("RAG settings: model=%s, db=%s, top_k=%d, provider=%s",
                 settings.embedding_model, settings.vector_db_dir,
                 settings.top_k, settings.llm_provider)
    return settings


def get_yaml_sources() -> dict[str, Any]:
    """Public accessor for the structured source config from rag_config.yaml.

    Loaders (this lesson) read their source-specific params (RSS URLs, dirs)
    through this so nothing hard-codes a feed list. Returns the 'sources'
    sub-mapping if present, else the whole file.
    """
    cfg = _load_yaml_config()
    return cfg.get("sources", cfg)
