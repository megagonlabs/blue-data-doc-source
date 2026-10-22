# blue-data-doc-api

FastAPI REST server for blue-data-doc-source. Exposes ingest, search, find, and delete as HTTP endpoints with interactive docs at `/docs`.

Full documentation — configuration, endpoints, and example requests — is in [platform/README.md](../README.md).

## Quick start

```bash
pip install -e lib/ -e platform/doc-api/
BLUE_DOC_API_KEY=choose-a-secret blue-data-doc-api
# → http://127.0.0.1:8001/docs
```

## Security & trust model

The REST API is network-facing and ingests server-side files, so it ships with the
shared `blue_data.server` policy (cross-repo XR-08) plus path confinement:

| Variable | Default | Effect |
|---|---|---|
| `BLUE_DOC_API_HOST` | `127.0.0.1` | Bind host — loopback only unless overridden |
| `BLUE_DOC_API_PORT` | `8001` | Bind port (family port map: registry 8000, doc 8001, web 8002) |
| `BLUE_DOC_API_KEY` | _(unset)_ | Required on every data route; send as `X-API-Key: <key>` or `Authorization: Bearer <key>`. Without a key configured, data routes return **401** |
| `BLUE_DOC_API_AUTH_DISABLED` | _(unset)_ | Set truthy (`1`/`true`/`yes`) to run with no API key (local/dev) |
| `BLUE_DOC_ROOT` | _(unset)_ | When set, `/documents/ingest` and `/documents/delete` accept only a **filename resolved inside this directory** (via blue-data's `contain()` — no separators, absolute paths, or `..`); anything else returns **400**. When unset, paths pass through, but the route still requires an API key |

State-changing routes (`POST`/`PUT`/`PATCH`/`DELETE`) require an `application/json`
body; form, multipart, or text bodies are refused with **415** (CSRF protection).
`/documents/health` and the docs routes (`/docs`, `/redoc`, `/openapi.json`) stay open.

Without this, any network peer could have the server read an arbitrary `*.txt`/`*.pdf`/
`*.docx` and read its text back via search (finding DOCSRC-07). Set `BLUE_DOC_API_KEY`,
keep the loopback bind unless behind a trusted proxy, and set `BLUE_DOC_ROOT` to confine
ingestion to a documents directory.

## Testing

The REST layer is a thin adapter over the library (see [ADR-012](../../design/13-decision-log.md)), so ingestion/search coverage lives in the library suite:

```bash
pytest lib/tests/unit/ -v
```

The server policy (API key / CSRF / loopback bind) and `BLUE_DOC_ROOT` path
confinement are covered by the REST layer's own tests:

```bash
pytest platform/doc-api/tests/ -v
```

Smoke-test the running service through its interactive docs at `http://127.0.0.1:8001/docs`.
