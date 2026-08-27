"""
StockM v1.0 - RAG Text Chunker
==============================

Stage 2 of the RAG pipeline.

Flow:
    DocumentLoader
        ↓
    Document
        ↓
    TextChunker
        ↓
    list[Chunk]
        ↓
    EmbeddingModel

This module is intentionally responsible ONLY for chunking.

It does not:
- generate embeddings
- access ChromaDB
- perform retrieval
- call an LLM
- modify the Document
- read environment variables directly

Configuration is injected through RAGSettings.
"""

from __future__ import annotations

import logging

from .config import RAGSettings
from .contracts import TextChunker
from .schemas import Chunk, Document


logger = logging.getLogger("stockm.rag.chunker")


class FixedSizeTextChunker(TextChunker):
    """
    Fixed-size text chunker for StockM financial documents.

    The chunker follows the existing StockM RAG contract:

        Document -> list[Chunk]

    Configuration:
        chunk_size:
            Maximum target size of a chunk in characters.

        chunk_overlap:
            Number of characters shared between consecutive chunks.

        min_chunk_size:
            Minimum number of characters required for a final chunk.

    The defaults come from RAGSettings and are NOT hard-coded here.
    """

    def __init__(self, settings: RAGSettings) -> None:
        self.settings = settings

        if settings.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if settings.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if settings.chunk_overlap >= settings.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        if settings.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be greater than 0")

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split one Document into ordered Chunks.

        Parameters
        ----------
        document:
            StockM Document produced by a DocumentLoader.

        Returns
        -------
        list[Chunk]
            Ordered chunks preserving the parent document metadata.

        Notes
        -----
        Chunk IDs are deterministic:

            {doc_id}::{index}

        This matches the Chunk schema and allows downstream vector-store
        implementations to remain idempotent.
        """

        text = document.text.strip()

        if not text:
            logger.debug(
                "Skipping empty document: doc_id=%s",
                document.doc_id,
            )
            return []

        chunks = self._split_text(text)

        result: list[Chunk] = []

        for index, chunk_text in enumerate(chunks):
            cleaned = chunk_text.strip()

            if not cleaned:
                continue

            chunk_id = f"{document.doc_id}::{index}"

            metadata = self._build_metadata(document)

            result.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    text=cleaned,
                    index=index,
                    metadata=metadata,
                )
            )

        logger.debug(
            "Chunked document: doc_id=%s chunks=%d",
            document.doc_id,
            len(result),
        )

        return result

    def _split_text(self, text: str) -> list[str]:
        """
        Split text using fixed-size character windows with overlap.

        The implementation prefers natural boundaries such as:

        1. paragraph/newline
        2. sentence boundary
        3. whitespace
        4. hard character boundary

        This keeps the implementation deterministic while preserving
        as much financial narrative context as possible.
        """

        chunk_size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap
        min_chunk_size = self.settings.min_chunk_size

        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            # If this is the final section, take the remainder.
            if end >= text_length:
                candidate = text[start:text_length].strip()

                if candidate:
                    chunks.append(candidate)

                break

            # Try to move the boundary to a natural location.
            boundary = self._find_boundary(
                text=text,
                start=start,
                end=end,
            )

            candidate = text[start:boundary].strip()

            if candidate:
                chunks.append(candidate)

            # Prevent an invalid or infinite step.
            if boundary <= start:
                boundary = end

            next_start = boundary - overlap

            # Ensure forward progress.
            if next_start <= start:
                next_start = boundary

            start = next_start

        # Remove very small trailing chunks.
        chunks = self._merge_small_final_chunk(chunks)

        # Final cleanup.
        return [
            chunk.strip()
            for chunk in chunks
            if chunk.strip()
        ]

    def _find_boundary(
        self,
        text: str,
        start: int,
        end: int,
    ) -> int:
        """
        Find a natural chunk boundary close to the target end.

        Preference:

        1. Newline
        2. Sentence-ending punctuation
        3. Whitespace
        4. Hard boundary

        A search window prevents the chunker from moving too far
        away from the configured chunk size.
        """

        minimum_boundary = start + max(
            1,
            int(self.settings.chunk_size * 0.50),
        )

        search_start = max(start, minimum_boundary)

        # ---------------------------------------------------------
        # 1. Prefer paragraph/newline boundary.
        # ---------------------------------------------------------

        newline_position = text.rfind("\n", search_start, end)

        if newline_position > start:
            return newline_position

        # ---------------------------------------------------------
        # 2. Prefer sentence boundary.
        # ---------------------------------------------------------

        sentence_boundaries = [".", "!", "?", "。", "！", "？"]

        best_sentence_position = -1

        for punctuation in sentence_boundaries:
            position = text.rfind(
                punctuation,
                search_start,
                end,
            )

            if position > best_sentence_position:
                best_sentence_position = position

        if best_sentence_position > start:
            return best_sentence_position + 1

        # ---------------------------------------------------------
        # 3. Prefer whitespace boundary.
        # ---------------------------------------------------------

        whitespace_position = text.rfind(
            " ",
            search_start,
            end,
        )

        if whitespace_position > start:
            return whitespace_position

        # ---------------------------------------------------------
        # 4. Fall back to hard character boundary.
        # ---------------------------------------------------------

        return end

    def _merge_small_final_chunk(
        self,
        chunks: list[str],
    ) -> list[str]:
        """
        Avoid leaving a useless tiny trailing chunk.

        If the final chunk is smaller than min_chunk_size, merge it
        into the previous chunk when possible.

        This is particularly useful for financial documents where
        a few leftover characters should not become an independent
        vector in the database.
        """

        if len(chunks) <= 1:
            return chunks

        minimum = self.settings.min_chunk_size

        final_chunk = chunks[-1]

        if len(final_chunk) >= minimum:
            return chunks

        previous_chunk = chunks[-2]

        merged = f"{previous_chunk} {final_chunk}".strip()

        chunks[-2] = merged
        chunks.pop()

        return chunks

    @staticmethod
    def _build_metadata(
        document: Document,
    ) -> dict[str, object]:
        """
        Propagate parent Document metadata onto every Chunk.

        These fields are important for future ChromaDB filtering:

        - company
        - category
        - source
        - url
        - published_at
        - language

        Additional source-specific metadata from Document.extra
        is preserved as well.
        """

        metadata: dict[str, object] = {
            "doc_id": document.doc_id,
            "source": document.source,
            "company": document.company,
            "category": document.category.value,
            "url": document.url,
            "published_at": (
                document.published_at.isoformat()
                if document.published_at is not None
                else None
            ),
            "language": document.language,
        }

        # Preserve source-specific metadata without modifying
        # the original Document.
        metadata.update(document.extra)

        return metadata


# ---------------------------------------------------------------------------
# Backward-friendly alias
# ---------------------------------------------------------------------------
#
# The abstract interface is called TextChunker.
# The concrete implementation is named FixedSizeTextChunker because
# the implementation strategy is fixed-size character chunking.
#
# Keeping the concrete name explicit makes future strategies possible:
#
#   SentenceTextChunker
#   SemanticTextChunker
#
# without changing the TextChunker contract.
# ---------------------------------------------------------------------------

Chunker = FixedSizeTextChunker