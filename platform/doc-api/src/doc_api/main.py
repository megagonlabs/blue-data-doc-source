"""blue-data-doc-source REST API.

Run:
    # install
    pip install -e ../../lib && pip install -e .

    # start (auto-reload for dev)
    uvicorn doc_api.main:app --reload --port 8001

    # or via the installed script
    blue-data-doc-api

Configure via environment variables:

    BLUE_DOC_EMBEDDING          openai | dummy  (default: dummy)
    OPENAI_API_KEY              required when BLUE_DOC_EMBEDDING=openai
    BLUE_DOC_REDIS_HOST         Redis host; omit to run without a vector store
    BLUE_DOC_REDIS_PORT         default: 6379
    BLUE_DOC_COLLECTION         default: documents
    BLUE_DOC_CHUNK_SIZE         default: 1000
    BLUE_DOC_OVERLAP            default: 150
    BLUE_DOC_OCR_FALLBACK       1 | true to enable Tesseract OCR on scanned PDFs

Server policy (via blue_data.server, cross-repo XR-08 / #75):

    BLUE_DOC_API_HOST           bind host; default 127.0.0.1 (loopback only)
    BLUE_DOC_API_PORT           bind port; default 8001 (family port map)
    BLUE_DOC_API_KEY            API key required on every data route (sent as
                                X-API-Key or Authorization: Bearer). Without it,
                                data routes return 401.
    BLUE_DOC_API_AUTH_DISABLED  set truthy (1/true/yes) to run with no API key
    BLUE_DOC_ROOT               when set, ingest/delete accept only a filename
                                resolved inside this directory (path confinement);
                                a network caller can no longer name an arbitrary
                                server file. See routers/documents.py.

State-changing routes (POST/PUT/PATCH/DELETE) require an application/json body;
form, multipart, or text bodies are refused with 415 (CSRF protection). The
health and docs paths (/documents/health, /docs, /redoc, /openapi.json) are open.
"""
from contextlib import asynccontextmanager

import uvicorn
from blue_data.server import DEFAULT_OPEN_PATHS, resolve_bind, security_dependencies
from fastapi import FastAPI

from doc_api.routers import documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    from doc_api.config import get_pipeline
    get_pipeline()
    yield


# /documents/health is the health probe; the default open paths cover the
# unprefixed docs routes. Everything else requires an API key and a JSON body.
_OPEN_PATHS = DEFAULT_OPEN_PATHS | {"/documents/health"}

app = FastAPI(
    title="blue-data-doc-source API",
    description="REST API for ingesting, searching, and managing documents.",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=security_dependencies("BLUE_DOC_API", open_paths=_OPEN_PATHS),
)

app.include_router(documents.router)


def run():
    # Loopback by default; host/port from BLUE_DOC_API_HOST / BLUE_DOC_API_PORT.
    host, port = resolve_bind("BLUE_DOC_API", default_port=8001)
    uvicorn.run("doc_api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
