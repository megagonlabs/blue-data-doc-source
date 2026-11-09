"""Pipeline singleton factory — reads configuration from environment variables.

Environment variables:

    BLUE_DOC_EMBEDDING          openai | dummy  (default: dummy)
    OPENAI_API_KEY              required when BLUE_DOC_EMBEDDING=openai
    OPENAI_EMBEDDING_MODEL      default: text-embedding-3-small
    BLUE_DOC_REDIS_URL          full redis://|rediss:// URL (host, auth, TLS, db); or use:
    BLUE_DOC_REDIS_HOST         Redis host; omit (and no URL) to run without a vector store
    BLUE_DOC_REDIS_PORT         default: 6379
    BLUE_DOC_REDIS_USERNAME     Redis ACL username (optional)
    BLUE_DOC_REDIS_PASSWORD     Redis password (optional)
    BLUE_DOC_REDIS_SSL          1 | true to connect over TLS
    BLUE_DOC_REDIS_PREFIX       key namespace / scan prefix; default: bluedoc
    BLUE_DOC_COLLECTION         default: documents
    BLUE_DOC_CHUNK_SIZE         default: 1000
    BLUE_DOC_OVERLAP            default: 150
    BLUE_DOC_OCR_FALLBACK       1 | true to enable Tesseract OCR on scanned PDFs
"""
from __future__ import annotations

import os
from typing import Optional

from blue_data_doc_source import (
    BasicEnricher,
    DocumentPipeline,
    DummyEmbeddingModel,
    KeywordClassifier,
    OpenAIEmbeddingModel,
    record_store_from_env,
)

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

    # Forward the whole Redis connection — URL, host/port, username, password, TLS — so a
    # password-protected or TLS Redis works, not just host/port (#195).
    from blue_data_doc_source import redis_vector_source_from_env
    vector_source = redis_vector_source_from_env("BLUE_DOC")

    ocr_fallback = os.getenv("BLUE_DOC_OCR_FALLBACK", "").lower() in ("1", "true", "yes")

    # Each CLI invocation is a fresh process, so the per-document records (written
    # at ingest when a classifier is set) must live in a *persistent* store or
    # `find`, `find-then-search`, `get-record` and cross-process `delete --path`
    # see nothing (#86). record_store_from_env persists them in the SAME Redis as the
    # chunks when one is configured, else a file under BLUE_DOC_RECORD_STORE (#188).
    return DocumentPipeline(
        chunk_size=int(os.getenv("BLUE_DOC_CHUNK_SIZE", "1000")),
        overlap=int(os.getenv("BLUE_DOC_OVERLAP", "150")),
        embedding_model=embedding_model,
        vector_source=vector_source,
        enricher=BasicEnricher(),
        classifier=KeywordClassifier(),
        record_store=record_store_from_env("BLUE_DOC"),
        collection=os.getenv("BLUE_DOC_COLLECTION", "documents"),
        ocr_fallback=ocr_fallback,
    )
