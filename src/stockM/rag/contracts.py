"""
StockM v1.0 - Phase 10, Lesson 2
Module Contracts (Interfaces / Abstract Base Classes)
=====================================================

Each RAG stage is defined here as an Abstract Base Class. Concrete
implementations (Lesson 3+) subclass these. The pipeline (Lesson 8) is wired
against the ABCs, NEVER against concrete classes - so swapping ChromaDB for
FAISS, or sentence-transformers for OpenAI embeddings, changes only the
object you inject, not the pipeline code.

Why ABCs and not typing.Protocol?
- ABCs give us a place to put shared helper logic and docstrings that concrete
  implementations inherit (DRY).
- ABCs enforce the contract at instantiation via @abstractmethod - a Protocol
  is structural-only and fails late. For an MLOps pipeline where a missing
  method would surface mid-batch, failing fast at construction is safer.

Each ABC declares exactly ONE responsibility (Single Responsibility Principle).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .schemas import Chunk, Document, RetrievedChunk


class DocumentLoader(ABC):
    """Stage 1: ingest raw financial documents from a source into Documents.

    A loader owns ONE source (a news API, an RSS feed, a filings directory).
    To support many sources, run many loaders and concatenate their outputs -
    do not build one mega-loader with source-switching branches (Open/Closed).
    """

    @abstractmethod
    def load(self, *args: Any, **kwargs: Any) -> list[Document]:
        """Return a batch of Documents. Empty list on no new content.

        Must be idempotent-friendly: callers may invoke load() repeatedly
        (e.g. daily cron); the loader should yield only new/changed docs or
        rely on downstream dedup (Lesson 4).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable source id, stored as Document.source (provenance)."""
        raise NotImplementedError


class TextChunker(ABC):
    """Stage 2: split a Document into retrieval-sized Chunks.

    The only contract: take a Document, return an ordered list of Chunks
    whose metadata is propagated from the parent. Chunking *strategy*
    (fixed-size, sentence, semantic) is an implementation detail (Lesson 4).
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split one Document into >=1 Chunks, preserving reading order."""
        raise NotImplementedError


class EmbeddingModel(ABC):
    """Stage 3: convert text into fixed-dim float vectors (embeddings).

    Embeddings are the *indexing* representation - they make semantic
    similarity computable. They are NOT the LLM; the LLM reasons over text,
    embeddings search over text. Two different jobs, two different models.

    Contract is batched (embed many texts in one call) because network/model
    calls dominate latency - one-at-a-time embedding is the #1 RAG perf bug.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality. The vector store needs this at creation."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier, stored as metadata for reproducibility (Lesson 13).

        Re-embedding with a different model produces incompatible vectors;
        recording the model name lets us detect and reindex when it changes.
        """
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts -> list of vectors, same length & order.

        Returns dense float vectors of length ``dimension`` each. Callers must
        not assume a specific normalization; similarity is normalized by the
        retriever, not here.
        """
        raise NotImplementedError


class VectorStore(ABC):
    """Stage 4: persist & search chunk embeddings + metadata.

    The store is the system of record for chunks. It must support CRUD plus
    metadata-filtered similarity search - that filtering is half the value of
    RAG in finance ("TCS earnings, last 30 days, exclude analyst reports").

    Concrete backends: ChromaDB (default, Lesson 6), FAISS, Pinecone. Each
    maps these methods to its own API; the pipeline never sees the backend.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert/update chunks and their vectors. Idempotent on chunk_id."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by id (e.g. retracted articles, GDPR/right-to-forget)."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return top-k chunks most similar to the query, optionally filtered.

        ``filters`` is a flat {metadata_key: value} dict. Backends translate
        it to their native filter syntax. Results are normalized to
        RetrievedChunk with score in [0,1] (higher = more relevant).
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Number of stored chunks. Used by /rag/status (Lesson 12) & tests."""
        raise NotImplementedError


class Retriever(ABC):
    """Stage 5: given a natural-language query, return ranked relevant chunks.

    The retriever OWNS the query->embedding step and the store->search call,
    plus any post-processing (reranking, dedup, score normalization). It is
    the single entry point the generator uses to fetch context.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Embed the query, search the store, return ranked RetrievedChunks."""
        raise NotImplementedError


class Generator(ABC):
    """Stage 6: produce a grounded answer from retrieved context + a query.

    The generator is where the LLM lives. Its contract is intentionally narrow:
    take the query + retrieved chunks, return a grounded answer PLUS the list
    of chunk_ids it used (for source attribution / XAI in Lesson 10).
    """

    @abstractmethod
    def generate(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
    ) -> "GenerationResult":
        """Return a grounded answer and the ids of chunks cited as evidence."""
        raise NotImplementedError


# Generator.generate returns GenerationResult. Imported at module bottom (not
# top) to avoid a theoretical import cycle: schemas.py is pure data and does
# not import contracts, but we keep the convention used elsewhere in StockM.
from .schemas import GenerationResult  # noqa: E402
