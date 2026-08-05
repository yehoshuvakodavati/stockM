"""
StockM v1.0 - Phase 10, Lesson 2
News Intelligence (RAG + Vector Database) - Package Root
=========================================================

This package implements StockM's Retrieval-Augmented Generation (RAG) layer.
It does NOT replace the quant prediction models (Phase 5/7/9); it sits
*alongside* them to explain, contextualize, and ground predictions in real
financial documents.

Pipeline (each stage is an interface/Protocol, swappable at runtime):

    raw docs -> loader -> chunker -> embeddings -> vector_store
                                                            |
    query -------------------------------------> retriever --+
                                                            |
                                              generator -> grounded answer + sources
                                                            ^
                                              pipeline orchestrates all stages

Design principles
-----------------
- SOLID: every module depends on a Protocol (interface), not a concrete class.
  Embedding models, vector DBs, and LLMs are all swappable with minimal change.
- Single responsibility per module.
- Configuration-driven (configs/rag_config.yaml), no hard-coded params.
- MLOps-friendly: every chunk/embedding carries provenance metadata so any
  answer can be traced back to its source document (reproducibility + XAI).

Public API: the high-level :class:`RAGPipeline` (Lesson 8) is the only object
callers should need. Submodules are exposed for advanced/injection use.
"""
from __future__ import annotations

# Re-export the core data structures so callers can do
# `from stockM.rag import Document, Chunk, RetrievedChunk` without reaching
# into private modules. Defined in Lesson 2; populated as lessons land.
from .schemas import Chunk, Document, GenerationResult, RetrievedChunk

__all__ = ["Document", "Chunk", "RetrievedChunk", "GenerationResult"]
