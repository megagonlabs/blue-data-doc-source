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
    # password-protected or TLS Redis works, not just host/port (#195). Back the record
    # store with the SAME Redis (or a file when no Redis), so FIND / record / path-delete
    # survive a restart instead of living only in memory while the chunks persist (#188).
    from blue_data_doc_source import record_store_from_env, redis_vector_source_from_env
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
        classifier=KeywordClassifier(),
        record_store=record_store_from_env("BLUE_DOC"),
        collection=os.getenv("BLUE_DOC_COLLECTION", "documents"),
        ocr_fallback=ocr_fallback,
    )
