# blue-data-doc-api

FastAPI REST server for blue-data-doc-source. Exposes ingest, search, find, and delete as HTTP endpoints with interactive docs at `/docs`.

Full documentation — configuration, endpoints, and example requests — is in [platform/README.md](../README.md).

## Quick start

```bash
pip install -e lib/ -e platform/doc-api/
blue-data-doc-api
# → http://localhost:8001/docs
```

## Testing

The REST layer is a thin adapter over the library (see [ADR-012](../../design/13-decision-log.md)), so ingestion/search coverage lives in the library suite:

```bash
pytest lib/tests/unit/ -v
```

Smoke-test the running service through its interactive docs at `http://localhost:8001/docs`.
