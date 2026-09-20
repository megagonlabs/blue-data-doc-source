"""blue-docdb — CLI for blue-data-doc-source.

Usage:
    blue-docdb [--format FORMAT] COMMAND [ARGS]...

Commands:
    ensure-index   Create the vector index for a collection
    ingest         Ingest a document into the vector store
    search         Search documents by semantic similarity
    delete         Delete all chunks for a document

Configure via environment variables:

    BLUE_DOC_EMBEDDING          openai | dummy  (default: dummy)
    OPENAI_API_KEY              required when BLUE_DOC_EMBEDDING=openai
    BLUE_DOC_REDIS_HOST         Redis host; omit to run without a vector store
    BLUE_DOC_COLLECTION         default collection name (default: documents)
    BLUE_DOC_CHUNK_SIZE         default: 1000
    BLUE_DOC_OVERLAP            default: 150
    BLUE_DOC_OCR_FALLBACK       1 | true to enable Tesseract OCR
"""
from __future__ import annotations

import json
import sys
from typing import Optional

import click

from blue_data_doc_cli.config import get_pipeline, reset_pipeline


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _emit(data: object, fmt: str) -> None:
    if fmt == "json":
        click.echo(json.dumps(data, indent=2, default=str))
    elif fmt == "nl":
        if isinstance(data, list):
            for item in data:
                click.echo(json.dumps(item, default=str))
        else:
            click.echo(json.dumps(data, default=str))
    else:
        if isinstance(data, list):
            for item in data:
                click.echo(_fmt_chunk(item) if isinstance(item, dict) else str(item))
        elif isinstance(data, dict):
            click.echo(_fmt_chunk(data))
        else:
            click.echo(str(data))


def _fmt_chunk(r: dict) -> str:
    parts = []
    header = r.get("chunk_id") or r.get("document_id") or ""
    file_name = r.get("file_name", "")
    if header:
        parts.append(f"[{header}]  {file_name}")
    section = r.get("section_title")
    page = r.get("page_start")
    meta: list[str] = []
    if section:
        meta.append(f"section: {section}")
    if page is not None:
        meta.append(f"page: {page}")
    if meta:
        parts.append("  " + "  |  ".join(meta))
    text = r.get("text", "")
    if text:
        preview = text[:200].replace("\n", " ")
        if len(text) > 200:
            preview += "…"
        parts.append(f"  {preview}")
    return "\n".join(parts)


def _err(msg: str) -> None:
    click.echo(f"Error: {msg}", err=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--format", "fmt",
    type=click.Choice(["table", "json", "nl"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format: table (default), json, or nl (newline-delimited JSON).",
)
@click.version_option(package_name="blue-data-doc-cli")
@click.pass_context
def cli(ctx: click.Context, fmt: str) -> None:
    """blue-docdb — ingest, search, and manage documents."""
    ctx.ensure_object(dict)
    ctx.obj["fmt"] = fmt


# ---------------------------------------------------------------------------
# ensure-index
# ---------------------------------------------------------------------------

@cli.command("ensure-index")
@click.option("--collection", default=None, help="Collection name (overrides BLUE_DOC_COLLECTION, default: documents).")
@click.pass_context
def ensure_index(ctx: click.Context, collection: Optional[str]) -> None:
    """Create the vector index for a collection.

    Must be called once before the first ingest when using a vector store.
    Safe to call multiple times — skips creation if the index already exists.
    """
    try:
        pipeline = get_pipeline()
        pipeline.ensure_index(collection=collection)
        coll = collection or pipeline.collection
    except Exception as exc:
        _err(str(exc))

    fmt = ctx.obj["fmt"]
    if fmt == "json":
        click.echo(json.dumps({"status": "ready", "collection": coll}))
    elif fmt == "nl":
        click.echo(json.dumps({"status": "ready", "collection": coll}))
    else:
        click.echo(f"Index ready: {coll}")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("path")
@click.option("--collection", default=None, help="Collection name (overrides BLUE_DOC_COLLECTION, default: documents).")
@click.pass_context
def ingest(ctx: click.Context, path: str, collection: Optional[str]) -> None:
    """Ingest a document into the vector store.

    Parses the file, splits it into section-aware chunks, extracts entities and
    keywords, generates embeddings, and upserts to the configured vector store.
    Re-ingesting the same file is safe — content-hash deduplication means the
    same file always produces the same chunk IDs.

    Supported: .txt, .pdf, .docx

    Prints document_id and chunk_count on success. Store these values to enable
    cross-session deletion with `blue-docdb delete --id`.
    """
    try:
        pipeline = get_pipeline()
        records = pipeline.ingest(path, collection=collection)
    except FileNotFoundError:
        _err(f"File not found: {path}")
    except ValueError as exc:
        _err(str(exc))
    except Exception as exc:
        _err(f"Ingest failed: {exc}")

    if not records:
        _err("No chunks produced — file may be empty or unparseable.")

    result = {
        "document_id": records[0]["document_id"],
        "file_name": records[0]["file_name"],
        "chunk_count": len(records),
        "collection": collection or pipeline.collection,
    }
    fmt = ctx.obj["fmt"]
    if fmt in ("json", "nl"):
        click.echo(json.dumps(result, indent=2 if fmt == "json" else None))
    else:
        click.echo(f"Ingested  {result['file_name']}")
        click.echo(f"  document_id:  {result['document_id']}")
        click.echo(f"  chunks:       {result['chunk_count']}")
        click.echo(f"  collection:   {result['collection']}")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("--collection", default=None, help="Collection name (overrides BLUE_DOC_COLLECTION, default: documents).")
@click.option("-k", default=5, show_default=True, help="Number of results to return.")
@click.pass_context
def search(ctx: click.Context, query: str, collection: Optional[str], k: int) -> None:
    """Search documents by semantic similarity.

    Embeds the query and returns the k most relevant chunks from the vector
    store. Results include file name, section title, page number (for PDFs),
    and a text preview.

    Requires BLUE_DOC_REDIS_HOST and an indexed collection.
    """
    try:
        pipeline = get_pipeline()
        results = pipeline.search(query, k=k, collection=collection)
    except ValueError as exc:
        _err(str(exc))
    except Exception as exc:
        _err(f"Search failed: {exc}")

    fmt = ctx.obj["fmt"]
    if fmt == "json":
        click.echo(json.dumps(results, indent=2, default=str))
        return
    if fmt == "nl":
        for r in results:
            click.echo(json.dumps(r, default=str))
        return

    if not results:
        click.echo("No results.")
        return

    col_w = {"file": 30, "section": 25, "page": 4, "preview": 60}
    header = (
        f"{'file':<{col_w['file']}}  {'section':<{col_w['section']}}  "
        f"{'pg':>{col_w['page']}}  {'text preview':<{col_w['preview']}}"
    )
    click.echo(header)
    click.echo("-" * len(header))
    for r in results:
        file_name = (r.get("file_name") or "")[:col_w["file"]]
        section = (r.get("section_title") or "")[:col_w["section"]]
        page = str(r["page_start"]) if r.get("page_start") is not None else "—"
        text = (r.get("text") or "").replace("\n", " ")[:col_w["preview"]]
        click.echo(
            f"{file_name:<{col_w['file']}}  {section:<{col_w['section']}}  "
            f"{page:>{col_w['page']}}  {text:<{col_w['preview']}}"
        )


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--path", "file_path", default=None, help="Path to the source file.")
@click.option("--id", "document_id", default=None, help="document_id from a previous ingest.")
@click.option("--chunks", "chunk_count", default=None, type=int,
              help="chunk_count from a previous ingest (required with --id).")
@click.option("--collection", default=None, help="Collection name (overrides BLUE_DOC_COLLECTION, default: documents).")
@click.pass_context
def delete(
    ctx: click.Context,
    file_path: Optional[str],
    document_id: Optional[str],
    chunk_count: Optional[int],
    collection: Optional[str],
) -> None:
    """Delete all chunks for a document from the vector store.

    Two ways to identify the document:

    \b
    By file path (file must still exist):
        blue-docdb delete --path reports/policy.pdf

    \b
    By document_id + chunk_count (cross-session, file may be gone):
        blue-docdb delete --id doc_3a7f1c2e --chunks 12

    document_id and chunk_count are printed by `blue-docdb ingest`.
    """
    if not file_path and not document_id:
        _err("Provide --path or --id + --chunks.")
    if document_id and chunk_count is None:
        _err("--chunks is required when using --id.")

    try:
        pipeline = get_pipeline()
        deleted = pipeline.delete(
            path=file_path,
            document_id=document_id,
            chunk_count=chunk_count,
            collection=collection,
        )
    except ValueError as exc:
        _err(str(exc))
    except Exception as exc:
        _err(f"Delete failed: {exc}")

    fmt = ctx.obj["fmt"]
    result = {"deleted": deleted, "document": file_path or document_id}
    if fmt == "json":
        click.echo(json.dumps(result))
    elif fmt == "nl":
        click.echo(json.dumps(result))
    else:
        label = file_path or document_id
        click.echo(f"Deleted {deleted} chunk(s) for {label}.")


if __name__ == "__main__":
    cli()
