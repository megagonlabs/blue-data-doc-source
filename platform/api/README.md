# blue-data-doc-api

FastAPI REST server for blue-data-doc-source. Exposes ingest, search, find, and delete as HTTP endpoints with interactive docs at `/docs`.

Full documentation — configuration, endpoints, and example requests — is in [platform/README.md](../README.md).

## Quick start

```bash
pip install -e lib/ -e platform/api/
blue-data-doc-api
# → http://localhost:8001/docs
```
