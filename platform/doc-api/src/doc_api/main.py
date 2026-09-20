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
"""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from doc_api.routers import documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    from doc_api.config import get_pipeline
    get_pipeline()
    yield


app = FastAPI(
    title="blue-data-doc-source API",
    description="REST API for ingesting, searching, and managing documents.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents.router)


def run():
    uvicorn.run("doc_api.main:app", host="0.0.0.0", port=8001, reload=False)


if __name__ == "__main__":
    run()
