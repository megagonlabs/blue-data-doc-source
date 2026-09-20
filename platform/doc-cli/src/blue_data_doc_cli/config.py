"""Pipeline singleton factory — reads configuration from environment variables.

Environment variables:

    BLUE_DOC_EMBEDDING          openai | dummy  (default: dummy)
    OPENAI_API_KEY              required when BLUE_DOC_EMBEDDING=openai
    OPENAI_EMBEDDING_MODEL      default: text-embedding-3-small
    BLUE_DOC_REDIS_HOST         Redis host; omit to run without a vector store
    BLUE_DOC_REDIS_PORT         default: 6379
    BLUE_DOC_COLLECTION         default: documents
    BLUE_DOC_CHUNK_SIZE         default: 1000
    BLUE_DOC_OVERLAP            default: 150
    BLUE_DOC_OCR_FALLBACK       1 | true to enable Tesseract OCR on scanned PDFs
"""
from __future__ import annotations

import os
from typing import Optional

from blue_data_doc_source import BasicEnricher, DocumentPipeline, DummyEmbeddingModel, OpenAIEmbeddingModel

_pipeline: Optional[DocumentPipeline] = None


def get_pipeline() -> DocumentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    _pipeline = None


def _build_pipeline() -> DocumentPipeline:
    embedding_backend = os.getenv("BLUE_DOC_EMBEDDING", "dummy").lower()
    if embedding_backend == "openai":
        embedding_model = OpenAIEmbeddingModel(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    else:
        embedding_model = DummyEmbeddingModel()

    vector_source = None
    redis_host = os.getenv("BLUE_DOC_REDIS_HOST")
    if redis_host:
        from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource
        vector_source = RedisVectorSource(
            name="redis",
            host=redis_host,
            port=int(os.getenv("BLUE_DOC_REDIS_PORT", "6379")),
        )

    ocr_fallback = os.getenv("BLUE_DOC_OCR_FALLBACK", "").lower() in ("1", "true", "yes")

    return DocumentPipeline(
        chunk_size=int(os.getenv("BLUE_DOC_CHUNK_SIZE", "1000")),
        overlap=int(os.getenv("BLUE_DOC_OVERLAP", "150")),
        embedding_model=embedding_model,
        vector_source=vector_source,
        enricher=BasicEnricher(),
        collection=os.getenv("BLUE_DOC_COLLECTION", "documents"),
        ocr_fallback=ocr_fallback,
    )
