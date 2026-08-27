"""
StockM v1.0 - RAG Embeddings
============================

Stage 3 of the RAG pipeline.

Flow:

    Document
        ↓
    Chunker
        ↓
    Chunk[]
        ↓
    EmbeddingModel
        ↓
    list[list[float]]
        ↓
    VectorStore / ChromaDB

This module is responsible ONLY for converting text into embeddings.

It does not:
- chunk documents
- store vectors
- retrieve documents
- generate answers
- call an external AI API

The implementation follows the existing StockM EmbeddingModel contract.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from .config import RAGSettings
from .contracts import EmbeddingModel


logger = logging.getLogger("stockm.rag.embeddings")


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """
    Sentence-Transformers implementation of the StockM EmbeddingModel contract.

    The model is loaded once when this object is created and reused for
    subsequent embedding requests.

    This is important because repeatedly loading the model would be
    extremely expensive.
    """

    def __init__(self, settings: RAGSettings) -> None:
        self.settings = settings

        self._model_name = settings.embedding_model

        logger.info(
            "Loading embedding model: %s",
            self._model_name,
        )

        self._model = SentenceTransformer(
            self._model_name
        )

        dimension = self._model.get_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                f"Could not determine embedding dimension "
                f"for model '{self._model_name}'."
            )

        self._dimension = int(dimension)

        logger.info(
            "Embedding model loaded: model=%s dimension=%d",
            self._model_name,
            self._dimension,
        )

    @property
    def dimension(self) -> int:
        """
        Return the dimensionality of generated vectors.

        For the default StockM model:

            sentence-transformers/all-MiniLM-L6-v2

        the dimension is 384.
        """

        return self._dimension

    @property
    def model_name(self) -> str:
        """
        Return the embedding model identifier.

        This is stored later with vector-store metadata so StockM
        can detect incompatible embeddings if the model changes.
        """

        return self._model_name

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Convert a batch of texts into dense float vectors.

        Parameters
        ----------
        texts:
            List of text strings to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text.

        Important:
            The order of the returned vectors is the same as the
            order of the input texts.

        Empty input returns an empty list.
        """

        if not texts:
            return []

        # Validate input before passing it to the model.
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"texts[{index}] must be a string, "
                    f"got {type(text).__name__}"
                )

        logger.debug(
            "Embedding batch: count=%d model=%s",
            len(texts),
            self._model_name,
        )

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        # Sentence Transformers returns a NumPy array for a batch.
        # Convert it into plain Python lists because the StockM
        # EmbeddingModel contract specifies list[list[float]].
        result = [
            [float(value) for value in vector]
            for vector in embeddings
        ]

        # Defensive validation.
        if len(result) != len(texts):
            raise RuntimeError(
                "Embedding model returned a different number of "
                "vectors than input texts."
            )

        for index, vector in enumerate(result):
            if len(vector) != self._dimension:
                raise RuntimeError(
                    f"Invalid embedding dimension at index {index}: "
                    f"expected {self._dimension}, "
                    f"got {len(vector)}."
                )

        logger.debug(
            "Embedding batch complete: count=%d dimension=%d",
            len(result),
            self._dimension,
        )

        return result