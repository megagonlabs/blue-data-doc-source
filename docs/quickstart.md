# Quick start

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e lib/
```

## Inspect chunks (no API key, no vector store)

```python
from blue_data_doc_source import DocumentPipeline

pipeline = DocumentPipeline(chunk_size=500, overlap=100)
records = pipeline.ingest("data/sample_policy.txt")

for r in records:
    print(r["chunk_id"], r.get("page_start"), r["text"][:80])
```

Supported file types: `.txt`, `.pdf`, `.docx`

## Full pipeline (OpenAI + Redis)

```bash
docker run -d -p 6379:6379 redis/redis-stack-server:latest
export OPENAI_API_KEY=sk-...
```

```python
from blue_data_doc_source import DocumentDataSource, OpenAIEmbeddingModel, BasicEnricher
from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource

source = DocumentDataSource(
    name="contracts",
    embedding_model=OpenAIEmbeddingModel(model="text-embedding-3-small"),
    vector_source=RedisVectorSource(name="my_redis", properties={"connection": {"host": "localhost", "port": 6379}}),
    enricher=BasicEnricher(),
    chunk_size=1000,
    overlap=150,
    default_collection="contracts",
)

source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "contracts"})

records = source.execute_query({
    "op": "DOCUMENT.INGEST",
    "path": "contract.pdf",
    "collection": "contracts",
})
print(f"Ingested {len(records)} chunks")

results = source.execute_query({
    "op": "DOCUMENT.SEARCH",
    "query": "termination clause",
    "collection": "contracts",
    "k": 5,
})
for r in results:
    print(r["chunk_id"], r["text"][:120])
```

## Next steps

- [Full feature list](features.md) — all ops, chunkers, enrichers, extractors
- [Guide](guide.md) — DataRegistry integration, NL routing, graph backend
- [Examples](../examples/README.md) — runnable demos
