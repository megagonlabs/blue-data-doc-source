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
        OpenAIEmbeddingModel,
    )

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
            properties={"connection": {
                "host": redis_host,
                "port": int(os.getenv("BLUE_DOC_REDIS_PORT", "6379")),
            }},
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
