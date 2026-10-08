# blue-data-doc-source (library)

Python library that turns documents into queryable blue-data sources. Handles parsing, chunking, enrichment, embedding, vector upsert, and graph indexing through a single `DocumentDataSource` entry point.

See the [root README](../README.md) for full documentation, installation instructions, and usage examples.

## Package

`blue-data-doc-source` — install with:

```bash
pip install -e lib/
```

Requires Python 3.10+.

## Testing

```bash
pip install -e lib/ pytest

# Unit tests — no external services required
pytest lib/tests/unit/ -v

# Integration tests — require Redis Stack on localhost:6379
docker run -p 6379:6379 redis/redis-stack-server:latest
pytest lib/tests/integration/ -v
```

Retrieval and pipeline benchmarks live under `lib/tests/benchmarks/`.
