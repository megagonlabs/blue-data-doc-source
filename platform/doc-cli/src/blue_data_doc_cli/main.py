"""blue-docdb — CLI for blue-data-doc-source.

Usage:
    blue-docdb [--format FORMAT] COMMAND [ARGS]...

Commands:
    doc ensure-index      Create the vector index for a collection
    doc ingest            Ingest a document into the vector store
    doc search            Search documents by semantic similarity
    doc find              Filter documents by extracted fields (no embedding)
    doc find-then-search  Filter documents, then search text within matched docs
    doc get-record        Retrieve the schema record for an ingested document
    doc delete            Delete all chunks for a document

Configure via environment variables:

    BLUE_DOC_EMBEDDING          openai | dummy  (default: dummy)
    OPENAI_API_KEY              required when BLUE_DOC_EMBEDDING=openai
    BLUE_DOC_REDIS_HOST         Redis host; omit to run without a vector store
    BLUE_DOC_COLLECTION         default collection name (default: documents)
    BLUE_DOC_CHUNK_SIZE         default: 1000
    BLUE_DOC_OVERLAP            default: 150
    BLUE_DOC_OCR_FALLBACK       1 | true to enable Tesseract OCR

Error codes:

    NOT_FOUND       File or resource does not exist
    INVALID_ARGS    Missing or incompatible arguments
    INGEST_FAILED   Parsing, chunking, or embedding error during ingest
    SEARCH_FAILED   Query execution error
    DELETE_FAILED   Chunk removal error
    CONFIG_ERROR    Pipeline configuration or index creation error
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

import click

from blue_data_doc_cli.config import get_pipeline, reset_pipeline
from blue_data_doc_cli.profile import _cli_home, profile_group, profile_store, resolver_for


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _parse_filter(filter_str: Optional[str], fmt: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON filter string; exit with INVALID_ARGS on bad JSON."""
    if not filter_str:
        return None
    try:
        return json.loads(filter_str)
    except json.JSONDecodeError as exc:
        _err("INVALID_ARGS", f"--filter is not valid JSON: {exc}", fmt)


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


def _fmt_record(r: dict) -> str:
    """Table rendering for a DocumentRecord."""
    lines = [f"document_id:    {r.get('document_id', '')}"]
    if r.get("file_name"):
        lines.append(f"file_name:      {r['file_name']}")
    if r.get("document_type"):
        lines.append(f"document_type:  {r['document_type']}")
    if r.get("confidence") is not None:
        lines.append(f"confidence:     {r['confidence']:.2f}")
    fields = r.get("fields") or {}
    if fields:
        lines.append("fields:")
        for k, v in fields.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _print_search_table(results: list) -> None:
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


def _err(code: str, msg: str, fmt: str = "table") -> None:
    if fmt in ("json", "nl"):
        click.echo(json.dumps({"error": {"code": code, "message": msg}}), err=True)
    else:
        click.echo(f"ERROR: {code} {msg}", err=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--profile", "profile_name",
    envvar="BLUE_DATA_PROFILE",
    default=None,
    help="Configuration profile. Overrides BLUE_DATA_PROFILE. "
         "Defaults to the selection in ~/.blue-docdb/.selected.profile.",
)
@click.option(
    "--format", "fmt",
    type=click.Choice(["table", "json", "nl"], case_sensitive=False),
    default=None,
    show_default=True,
    help="Output format: table (default), json, or nl (newline-delimited JSON). "
         "Overrides BLUE_DATA_CLI_FORMAT.",
)
@click.version_option(package_name="blue-data-doc-cli")
@click.pass_context
def cli(ctx: click.Context, profile_name: Optional[str], fmt: Optional[str]) -> None:
    """blue-docdb — ingest, search, and manage documents."""
    home = _cli_home()
    profile_store(home).initialize()
    resolver = resolver_for(home, profile_name)
    resolver.export()

    resolved_fmt = fmt or resolver.get("BLUE_DATA_CLI_FORMAT") or "table"

    ctx.ensure_object(dict)
    ctx.obj["fmt"] = resolved_fmt
    ctx.obj["home"] = home
    ctx.obj["profile"] = resolver.profile
    ctx.obj["config"] = resolver


cli.add_command(profile_group)


# ---------------------------------------------------------------------------
# doc component group
# ---------------------------------------------------------------------------

@cli.group("doc")
@click.pass_context
def doc_group(ctx: click.Context) -> None:
    """Manage documents — ingest, search, find, and delete."""
    ctx.ensure_object(dict)


# ---------------------------------------------------------------------------
# doc ensure-index
# ---------------------------------------------------------------------------

@doc_group.command("ensure-index")
@click.option("--collection", default=None, help="Collection name (overrides BLUE_DOC_COLLECTION, default: documents).")
@click.pass_context
def ensure_index(ctx: click.Context, collection: Optional[str]) -> None:
    """Create the vector index for a collection.

    Must be called once before the first ingest when using a vector store.
    Safe to call multiple times — skips creation if the index already exists.

    \b
    Examples:
        blue-docdb doc ensure-index
        blue-docdb doc ensure-index --collection contracts
        blue-docdb --format json doc ensure-index --collection contracts
    """
    fmt = ctx.obj["fmt"]
    try:
        pipeline = get_pipeline()
        pipeline.ensure_index(collection=collection)
        coll = collection or pipeline.collection
    except Exception as exc:
        _err("CONFIG_ERROR", str(exc), fmt)

    if fmt == "json":
        click.echo(json.dumps({"status": "ready", "collection": coll}))
    elif fmt == "nl":
        click.echo(json.dumps({"status": "ready", "collection": coll}))
    else:
        click.echo(f"Index ready: {coll}")


# ---------------------------------------------------------------------------
# doc ingest
# ---------------------------------------------------------------------------

@doc_group.command("ingest")
@click.argument("path")
@click.option("--collection", default=None, help="Collection name (overrides BLUE_DOC_COLLECTION, default: documents).")
@click.pass_context
def ingest(ctx: click.Context, path: str, collection: Optional[str]) -> None:
    """Ingest a document into the vector store.

    Parses the file, splits it into section-aware chunks, extracts entities and
    keywords, and generates embeddings. When BLUE_DOC_REDIS_HOST is set, chunks
    are upserted to the vector store; without it, ingest still runs and prints
    results but nothing is persisted for search.

    Re-ingesting the same file is safe — content-hash deduplication means the
    same file always produces the same chunk IDs.

    Supported: .txt, .pdf, .docx

    Prints document_id and chunk_count on success. Store these values to enable
    cross-session deletion with `blue-docdb doc delete --id`.

    \b
    Examples:
        blue-docdb doc ingest report.pdf
        blue-docdb doc ingest contract.docx --collection contracts
        blue-docdb --format json doc ingest report.pdf
    """
    fmt = ctx.obj["fmt"]
    try:
        pipeline = get_pipeline()
        records = pipeline.ingest(path, collection=collection)
    except FileNotFoundError:
        _err("NOT_FOUND", f"File not found: {path}", fmt)
    except ValueError as exc:
        _err("INVALID_ARGS", str(exc), fmt)
    except Exception as exc:
        _err("INGEST_FAILED", str(exc), fmt)

    if not records:
        _err("INGEST_FAILED", "No chunks produced — file may be empty or unparseable.", fmt)

    result = {
        "document_id": records[0]["document_id"],
        "file_name": records[0]["file_name"],
        "chunk_count": len(records),
        "collection": collection or pipeline.collection,
    }
    if fmt in ("json", "nl"):
        click.echo(json.dumps(result, indent=2 if fmt == "json" else None))
    else:
        click.echo(f"Ingested  {result['file_name']}")
        click.echo(f"  document_id:  {result['document_id']}")
        click.echo(f"  chunks:       {result['chunk_count']}")
        click.echo(f"  collection:   {result['collection']}")


# ---------------------------------------------------------------------------
# doc search
# ---------------------------------------------------------------------------

@doc_group.command("search")
@click.argument("query")
@click.option("--collection", default=None, help="Collection name (overrides BLUE_DOC_COLLECTION, default: documents).")
@click.option("-k", default=5, show_default=True, help="Number of results to return.")
@click.option(
    "--mode",
    type=click.Choice(["vector", "hybrid", "bm25", "rerank"], case_sensitive=False),
    default="vector",
    show_default=True,
    help="Search mode.",
)
@click.option("--filter", "filter_str", default=None, help="JSON filter applied before vector search, e.g. '{\"document_type\":\"contract\"}'.")
@click.option("--context-chunks", default=1, show_default=True, help="Adjacent chunks to include around each result.")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    collection: Optional[str],
    k: int,
    mode: str,
    filter_str: Optional[str],
    context_chunks: int,
) -> None:
    """Search documents by semantic similarity.

    Embeds the query and returns the k most relevant chunks from the vector
    store. Results include file name, section title, page number (for PDFs),
    and a text preview.

    Requires BLUE_DOC_REDIS_HOST and an indexed collection.

    \b
    Examples:
        blue-docdb doc search "termination clause"
        blue-docdb doc search "governing law" -k 10
        blue-docdb doc search "deductible" --collection policies
        blue-docdb doc search "liability" --mode hybrid
        blue-docdb doc search "payment terms" --filter '{"document_type":"contract"}'
        blue-docdb --format json doc search "payment terms" | jq '.[0].text'
        blue-docdb --format nl doc search "eligibility" | jq -r '.text'
    """
    fmt = ctx.obj["fmt"]
    filters = _parse_filter(filter_str, fmt)
    try:
        pipeline = get_pipeline()
        results = pipeline.search(
            query, k=k, collection=collection,
            mode=mode, filters=filters, context_chunks=context_chunks,
        )
    except ValueError as exc:
        _err("INVALID_ARGS", str(exc), fmt)
    except Exception as exc:
        _err("SEARCH_FAILED", str(exc), fmt)

    if fmt == "json":
        click.echo(json.dumps(results, indent=2, default=str))
        return
    if fmt == "nl":
        for r in results:
            click.echo(json.dumps(r, default=str))
        return

    _print_search_table(results)


# ---------------------------------------------------------------------------
# doc find
# ---------------------------------------------------------------------------

@doc_group.command("find")
@click.option("--filter", "filter_str", default=None, help="JSON filter dict, e.g. '{\"document_type\":\"contract\"}'.")
@click.option("--collection", default=None, help="Collection name (overrides BLUE_DOC_COLLECTION, default: documents).")
@click.pass_context
def find(ctx: click.Context, filter_str: Optional[str], collection: Optional[str]) -> None:
    """Filter documents by extracted fields — no embedding, no LLM.

    Returns DocumentRecords matching the filter. Each record holds the
    document_id, document_type, confidence score, and any fields extracted
    at ingest time (parties, governing_law, total_due, etc.).

    Records are only populated when a classifier is configured at pipeline
    construction. Without a classifier, find always returns an empty list.

    \b
    Filter syntax (all conditions ANDed):
        {"document_type": "contract"}
        {"fields.total_due": {"gt": 10000}}
        {"fields.vendor": {"contains": "Acme"}}

    \b
    Examples:
        blue-docdb doc find
        blue-docdb doc find --filter '{"document_type":"contract"}'
        blue-docdb doc find --filter '{"fields.total_due":{"gt":50000}}'
        blue-docdb --format json doc find --filter '{"document_type":"invoice"}'
    """
    fmt = ctx.obj["fmt"]
    filters = _parse_filter(filter_str, fmt)
    try:
        pipeline = get_pipeline()
        records = pipeline.find_records(filters)
    except Exception as exc:
        _err("SEARCH_FAILED", str(exc), fmt)

    if fmt == "json":
        click.echo(json.dumps(records, indent=2, default=str))
        return
    if fmt == "nl":
        for r in records:
            click.echo(json.dumps(r, default=str))
        return

    if not records:
        click.echo("No records found.")
        return
    for r in records:
        click.echo(_fmt_record(r))
        click.echo("")


# ---------------------------------------------------------------------------
# doc find-then-search
# ---------------------------------------------------------------------------

@doc_group.command("find-then-search")
@click.argument("query")
@click.option("--filter", "filter_str", default=None, help="JSON filter to scope documents before searching.")
@click.option("--collection", default=None, help="Collection name (overrides BLUE_DOC_COLLECTION, default: documents).")
@click.option("-k", default=5, show_default=True, help="Number of results to return.")
@click.option(
    "--mode",
    type=click.Choice(["vector", "hybrid", "bm25", "rerank"], case_sensitive=False),
    default="vector",
    show_default=True,
    help="Search mode.",
)
@click.pass_context
def find_then_search(
    ctx: click.Context,
    query: str,
    filter_str: Optional[str],
    collection: Optional[str],
    k: int,
    mode: str,
) -> None:
    """Filter documents by fields, then search text within matched docs.

    Two-step operation: first filters DocumentRecords by --filter (same syntax
    as `find`), then runs a semantic search scoped to those documents only.
    Results are merged and ranked by score across all matched documents.

    Useful when you need both structured filtering (document_type, fields) and
    semantic retrieval in a single call.

    \b
    Examples:
        blue-docdb doc find-then-search "termination clause" --filter '{"document_type":"contract"}'
        blue-docdb doc find-then-search "deductible" --filter '{"document_type":"policy"}' -k 3
        blue-docdb --format json doc find-then-search "payment" --filter '{"fields.total_due":{"gt":1000}}'
    """
    fmt = ctx.obj["fmt"]
    filters = _parse_filter(filter_str, fmt)
    try:
        pipeline = get_pipeline()
        matched_docs = pipeline.find_records(filters)
        if not matched_docs:
            results = []
        else:
            doc_ids = [r["document_id"] for r in matched_docs]
            all_results: list = []
            seen: set = set()
            for doc_id in doc_ids:
                doc_results = pipeline.search(
                    query, k=k, collection=collection,
                    filters={"document_id": doc_id}, mode=mode,
                )
                for r in doc_results:
                    cid = r.get("chunk_id")
                    if cid not in seen:
                        seen.add(cid)
                        all_results.append(r)
            results = sorted(all_results, key=lambda r: r.get("score", 0))[:k]
    except ValueError as exc:
        _err("INVALID_ARGS", str(exc), fmt)
    except Exception as exc:
        _err("SEARCH_FAILED", str(exc), fmt)

    if fmt == "json":
        click.echo(json.dumps(results, indent=2, default=str))
        return
    if fmt == "nl":
        for r in results:
            click.echo(json.dumps(r, default=str))
        return

    _print_search_table(results)


# ---------------------------------------------------------------------------
# doc get-record
# ---------------------------------------------------------------------------

@doc_group.command("get-record")
@click.option("--document-id", required=True, help="document_id from a previous ingest.")
@click.pass_context
def get_record(ctx: click.Context, document_id: str) -> None:
    """Retrieve the schema record for an ingested document.

    Returns the DocumentRecord written at ingest time: document_type,
    confidence, classifier used, and any extracted fields (parties,
    governing_law, total_due, etc.).

    Records are only available when a classifier was configured at pipeline
    construction. Returns NOT_FOUND if no record exists for the document_id.

    \b
    Examples:
        blue-docdb doc get-record --document-id doc_3a7f1c2e
        blue-docdb --format json doc get-record --document-id doc_3a7f1c2e
    """
    fmt = ctx.obj["fmt"]
    try:
        pipeline = get_pipeline()
        record = pipeline.get_record(document_id)
    except Exception as exc:
        _err("SEARCH_FAILED", str(exc), fmt)

    if record is None:
        _err("NOT_FOUND", f"No record found for document_id: {document_id}", fmt)

    if fmt == "json":
        click.echo(json.dumps(record, indent=2, default=str))
        return
    if fmt == "nl":
        click.echo(json.dumps(record, default=str))
        return

    click.echo(_fmt_record(record))


# ---------------------------------------------------------------------------
# doc delete
# ---------------------------------------------------------------------------

@doc_group.command("delete")
@click.option("--path", "file_path", default=None, help="Path to the source file.")
@click.option("--id", "document_id", default=None, help="document_id from a previous ingest.")
@click.option("--chunks", "chunk_count", default=None, type=int,
              help="chunk_count from a previous ingest (required with --id).")
@click.option("--collection", default=None, help="Collection name (overrides BLUE_DOC_COLLECTION, default: documents).")
@click.option("--confirm", "confirmed", is_flag=True, default=False,
              help="Skip the confirmation prompt.")
@click.pass_context
def delete(
    ctx: click.Context,
    file_path: Optional[str],
    document_id: Optional[str],
    chunk_count: Optional[int],
    collection: Optional[str],
    confirmed: bool,
) -> None:
    """Delete all chunks for a document from the vector store.

    Requires BLUE_DOC_REDIS_HOST and an indexed collection.

    Two ways to identify the document:

    \b
    By file path (file must still exist):
        blue-docdb doc delete --path reports/policy.pdf --confirm

    \b
    By document_id + chunk_count (cross-session, file may be gone):
        blue-docdb doc delete --id doc_3a7f1c2e --chunks 12 --confirm

    document_id and chunk_count are printed by `blue-docdb doc ingest`.
    Omit --confirm to be prompted interactively before deletion.
    """
    fmt = ctx.obj["fmt"]
    if not file_path and not document_id:
        _err("INVALID_ARGS", "Provide --path or --id + --chunks.", fmt)
    if document_id and chunk_count is None:
        _err("INVALID_ARGS", "--chunks is required when using --id.", fmt)

    label = file_path or document_id
    if not confirmed:
        click.confirm(f"Delete all chunks for '{label}'?", abort=True)

    try:
        pipeline = get_pipeline()
        deleted = pipeline.delete(
            path=file_path,
            document_id=document_id,
            chunk_count=chunk_count,
            collection=collection,
        )
    except ValueError as exc:
        _err("INVALID_ARGS", str(exc), fmt)
    except Exception as exc:
        _err("DELETE_FAILED", str(exc), fmt)

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
