"""
StockM v1.0 - Phase 10, Lesson 2
Shared Data Schemas (the RAG vocabulary)
=========================================

Every module in the RAG pipeline exchanges data through these typed structures.
Defining them ONCE here means loader/chunker/embedder/retriever/generator never
disagree on what a "document" or a "chunk" is. This is the shared contract.

Why Pydantic (not dataclasses) here?
- Validates types at construction (catch a bad metadata dict before it poison-
  ing the vector DB).
- Serializes to/from JSON for the API layer (Lesson 12) and for persistence
  (Lesson 13) with zero glue code.
- ``model_config = {"frozen": ...}`` gives us hashable, immutable records once
  a chunk is stored - matching the immutability contract used in the Phase 9
  API settings (src/api/config.py).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentCategory(str, Enum):
    """The kind of financial document. Drives metadata filtering at retrieval.

    Filtering by category is how a user asks "only earnings reports" or
    "exclude analyst opinion". Treating category as an Enum (not a free
    string) prevents typo-driven silent misses (e.g. "earning" vs "earnings").
    """

    FINANCIAL_NEWS = "financial_news"
    COMPANY_ANNOUNCEMENT = "company_announcement"
    EARNINGS_REPORT = "earnings_report"
    PRESS_RELEASE = "press_release"
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_REPORT = "quarterly_report"
    ANALYST_REPORT = "analyst_report"
    ECONOMIC_NEWS = "economic_news"
    UNKNOWN = "unknown"


class Document(BaseModel):
    """A single ingested financial document, pre-chunking.

    This is what the loader produces and the chunker consumes. It carries the
    full text plus rich metadata - metadata that propagates onto every Chunk
    so retrieval can filter by company/date/category at query time.

    Attributes
    ----------
    doc_id : str
        Stable unique id (e.g. a hash of URL+content, or a source-provided id).
        Used as the key in the vector DB and for deduplication (Lesson 4).
    text : str
        Full cleaned document text. May be long (thousands of words); that's
        fine - the chunker will split it.
    source : str
        Origin publisher/feed (e.g. "moneycontrol", "bse_india", "reuters").
    url : str | None
        Canonical link back to the original - required for source attribution
        (Lesson 10 XAI). None only for private/internal docs.
    company : str | None
        Normalized ticker or company name this doc is about (e.g. "RELIANCE").
        None for macro/economic news. Drives per-company filtering.
    category : DocumentCategory
        What kind of document (see enum above).
    published_at : datetime | None
        When the event/doc was published (UTC). Critical: financial relevance
        is time-decaying, so we always store the original publish time, not
        the ingestion time. The retriever can filter "last 7 days".
    language : str
        ISO 639-1 code (default "en"). Supports future multi-lingual Indian
        press without schema changes.
    extra : dict[str, Any]
        Escape hatch for source-specific fields (author, sentiment label,
        section, etc.) without bloating the core schema.
    """

    model_config = {"frozen": True}

    doc_id: str
    text: str
    source: str
    url: str | None = None
    company: str | None = None
    category: DocumentCategory = DocumentCategory.UNKNOWN
    published_at: datetime | None = None
    language: str = "en"
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrieval-sized piece of a Document, ready to be embedded & stored.

    This is the unit of retrieval: we search over CHUNKS, not whole documents.
    A long earnings report becomes many chunks; only the relevant ones return.

    Attributes
    ----------
    chunk_id : str
        Stable unique id, typically "{doc_id}::{index}" - deterministic so
        re-running ingestion doesn't duplicate (Lesson 13 idempotency).
    doc_id : str
        Foreign key back to the parent Document - lets the generator group
        multiple chunks from the same article and cite it once.
    text : str
        The chunk's text. Sized by the chunker (Lesson 4) - usually 300-500
        tokens with overlap.
    index : int
        Position of this chunk within its parent (0-based). Preserves reading
        order so the generator can reconstruct narrative flow if needed.
    metadata : dict[str, Any]
        Propagated from the parent Document (company, category, source, url,
        published_at, language). This is what ChromaDB filters on at query
        time. Kept as a dict because vector-DB filter syntax varies by backend.
    """

    model_config = {"frozen": True}

    chunk_id: str
    doc_id: str
    text: str
    index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(Chunk):
    """A Chunk that came back from a search, with a relevance score.

    The score semantics depend on the vector store (cosine sim, L2 distance,
    etc.); the retriever (Lesson 7) normalizes it to a 0..1 "higher is more
    relevant" float so downstream code never branches on the backend.

    Attributes
    ----------
    score : float
        Normalized relevance in [0, 1]. 1.0 = identical to the query.
    """

    # Re-opens mutability only for the score, which the retriever sets after
    # the store returns results. We keep the chunk text/metadata immutable.
    model_config = {"frozen": False}

    score: float = 0.0


class GenerationResult(BaseModel):
    """The output of the RAG generator: a grounded answer + its evidence.

    This is what flows back to the API (Lesson 12) and to the explainable-AI
    layer (Lesson 10). Carrying ``cited_chunk_ids`` alongside the answer is
    what makes the system *auditable*: a reviewer can pull the exact chunks
    the LLM was allowed to see.

    Attributes
    ----------
    answer : str
        The natural-language answer, grounded in ``retrieved``.
    cited_chunk_ids : list[str]
        Ids of chunks the generator actually used as evidence (a subset of the
        retrieved chunks - the generator may drop irrelevant ones).
    model : str
        LLM identifier used (reproducibility - Lesson 13).
    prompt : str | None
        The full rendered prompt, for debugging/XAI inspection. None in
        production if prompt logging is disabled.
    """

    model_config = {"frozen": True}

    answer: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    model: str = ""
    prompt: str | None = None


def utc_now() -> datetime:
    """Timezone-aware UTC now. Centralized so tests can monkeypatch it.

    Why UTC everywhere: financial docs cross timezones (US market close, Asian
    open). Storing naive local time is a classic bug source. We normalize to
    UTC at ingest and format for display only at the API edge.
    """
    return datetime.now(timezone.utc)
