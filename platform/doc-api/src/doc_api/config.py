"""Pipeline singleton — built once from environment variables, reused across requests."""
import os

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


def _build_pipeline():
    from blue_data_doc_source import (
        BasicEnricher,
        DocumentPipeline,
        DummyEmbeddingModel,
        KeywordClassifier,
        OpenAIEmbeddingModel,
    )

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

    return DocumentPipeline(
        chunk_size=int(os.getenv("BLUE_DOC_CHUNK_SIZE", "1000")),
        overlap=int(os.getenv("BLUE_DOC_OVERLAP", "150")),
        embedding_model=embedding_model,
        vector_source=vector_source,
        enricher=BasicEnricher(),
        # Zero-API-cost classification so a DocumentRecord is written per document;
        # without it FIND / FIND_THEN_SEARCH / GET_RECORD always return empty (#86).
        # The API is a long-running process, so the auto-created MemoryRecordStore
        # persists records for the server's lifetime (same as the MCP server).
        classifier=KeywordClassifier(),
        collection=os.getenv("BLUE_DOC_COLLECTION", "documents"),
        ocr_fallback=ocr_fallback,
    )
