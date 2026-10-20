# blue-data-doc-source — Repo Organization

## Directory Tree

```
blue-data-doc-source/
│
├── README.md                            # project overview, quick start, package table
├── CLAUDE.md                            # Claude Code instructions for this repo
├── pyproject.toml                       # root package metadata and tool config (ruff, pytest)
│
├── design/                              # contributor-facing design documents (ADRs, architecture)
│
├── docs/                                # user-facing documentation
│   ├── architecture.md                  # pipeline layers and component responsibilities
│   ├── benchmarks.md                    # chunker and retrieval benchmark results
│   ├── chunkers.md                      # SectionChunker / WindowChunker / SentenceChunker reference
│   ├── extending.md                     # extension points: custom parsers, chunkers, enrichers
│   ├── features.md                      # feature-by-feature documentation with code examples
│   ├── guide.md                         # DataRegistry integration, NL routing, entry point guide
│   ├── installation.md                  # install instructions, extras, external deps
│   ├── mcp.md                           # MCP server setup and tool reference
│   ├── parsers.md                       # parser comparison: TextParser, PDFParser, PlumberPDFParser, DocxParser
│   └── quickstart.md                    # minimal install + first ingest in under 5 minutes
│
├── data/                                # fixture documents for integration tests and examples
│   ├── sample.txt
│   ├── sample_policy.txt / .pdf / .docx
│   ├── sample_contract.pdf
│   ├── sample_invoice.pdf
│   ├── sample_research.pdf
│   ├── real_annual_report.pdf           # real third-party docs — NOT committed; fetched
│   ├── real_research_paper.pdf          #   on demand via scripts/fetch_fixtures.py
│   │                                    #   (git-ignored; see README "Disclosures")
│   └── _generate_*.py                   # scripts that generated the sample_* fixture PDFs
│
├── examples/                            # runnable demos (most need no API key)
│   ├── README.md
│   ├── local_memory_store.py            # MemoryVectorSource + DummyEmbeddingModel, offline
│   ├── local_redis_store.py             # RedisVectorSource as persistent backend
│   ├── backends/
│   │   ├── ingest_txt.py                # minimal CLI-style ingest script
│   │   ├── memory_quickstart.py         # fastest path: no Redis, no API key
│   │   ├── openai_embeddings_demo.py    # semantic quality: OpenAI vs DummyEmbeddingModel
│   │   └── redis_backend_demo.py        # production-grade Redis backend
│   ├── feature-demos/
│   │   ├── chunking_demo.py             # SectionChunker vs WindowChunker on same text
│   │   ├── content_addressing_demo.py   # SHA-256 doc ID, idempotent re-ingest
│   │   ├── document_find_demo.py        # DOCUMENT.FIND structured field filters
│   │   ├── enrichment_demo.py           # BasicEnricher: entities + semantic_hints
│   │   ├── field_extraction_demo.py     # KeywordClassifier + LLMFieldExtractor
│   │   ├── find_then_search_demo.py     # DOCUMENT.FIND_THEN_SEARCH
│   │   ├── generic_extraction_demo.py   # StructuredFieldExtractor without classifier
│   │   ├── graph_backend_demo.py        # GraphIndexer: entity traversal and queries
│   │   ├── nl_query_demo.py             # DOCUMENT.SEARCH with natural language
│   │   ├── parsers_demo.py              # all four parsers on the same document
│   │   ├── pdf_formats_demo.py          # PDFParser on various PDF layouts
│   │   ├── rich_search_demo.py          # DOCUMENT.SEARCH with rich=True
│   │   └── structured_extractor_demo.py # StructuredFieldExtractor with Pydantic model
│   ├── framework-demos/
│   │   ├── langgraph_rag_demo.py        # two-node RAG graph: retrieve → generate
│   │   └── mcp_demo.py                  # LLM agent calling document-mcp tools
│   ├── integration-demos/
│   │   ├── corpus_update_demo.py        # re-ingest a changed document; dedup in action
│   │   ├── custom_parser_demo.py        # plug in a custom Parser subclass
│   │   ├── data_registry_demo.py        # DocumentDataSource registered in DataRegistry
│   │   ├── multi_format_demo.py         # ingest txt + pdf + docx in one pipeline
│   │   ├── pdf_pipeline_demo.py         # full PDF ingest + search flow
│   │   └── pdf_redis_demo.py            # PDF ingest to Redis vector store
│   └── scenario-demos/
│       ├── graph_value_demo.py          # graph traversal: entity → document → sections
│       ├── multi_document_demo.py       # corpus search across multiple docs
│       ├── provenance_demo.py           # page_start / page_end / section_title on results
│       ├── section_vs_window_demo.py    # SectionChunker vs WindowChunker retrieval quality
│       ├── semantic_search_demo.py      # embedding quality comparison
│       └── structured_doc_query_demo.py # classify → extract → find → search pipeline
│
├── lib/                                 # Python library (blue-data-doc-source PyPI package)
│   ├── README.md                        # package overview; defers to root README
│   ├── src/
│   │   └── blue_data_doc_source/
│   │       ├── __init__.py              # re-exports DocumentDataSource, DocumentPipeline, ChunkRecord, all ABCs, concrete classes
│   │       └── source/
│   │           └── document/
│   │               ├── document_source.py   # DocumentDataSource — public entry point; op dispatch
│   │               ├── pipeline.py          # DocumentPipeline — orchestrates the full ingest chain
│   │               ├── chunk_record.py      # ChunkRecord dataclass — canonical chunk schema with to_dict()
│   │               ├── nl_parser.py         # NLQueryParser — OpenAI JSON mode → DOCUMENT.* op dict
│   │               ├── parsers/
│   │               │   ├── base.py              # Parser ABC — parse(path) → list[page_dict]; parse_blocks() for typed content
│   │               │   ├── text_parser.py       # TextParser — plain .txt files
│   │               │   ├── pdf_parser.py        # PDFParser — pypdf; falls back to OCR via pytesseract
│   │               │   ├── plumber_pdf_parser.py # PlumberPDFParser — pdfplumber; table-aware extraction
│   │               │   └── docx_parser.py       # DocxParser — python-docx; preserves heading structure
│   │               ├── chunkers/
│   │               │   ├── base.py              # Chunker ABC — chunk_with_titles() → list[(section_title, text)]
│   │               │   ├── section_chunker.py   # SectionChunker — header-aware; word-count sized; default
│   │               │   ├── window_chunker.py    # WindowChunker — sliding word-count window with overlap
│   │               │   └── sentence_chunker.py  # SentenceChunker — sentence-boundary aware
│   │               ├── embeddings/
│   │               │   ├── base.py              # EmbeddingModel ABC — dim property + embed_text(str) → list[float]
│   │               │   ├── openai_embedding.py  # OpenAIEmbeddingModel — text-embedding-3-small (dim=1536)
│   │               │   └── dummy_embedding.py   # DummyEmbeddingModel — SHA-256 hash; deterministic; for tests
│   │               ├── enrichment/
│   │               │   ├── base.py              # Enricher ABC — enrich(text) → {entities, semantic_hints}
│   │               │   ├── basic_enricher.py    # BasicEnricher — regex PROPER_NOUN + semantic hint patterns; no ML
│   │               │   └── llm_enricher.py      # LLMEnricher — one GPT-4o-mini call/chunk; keyphrases + relationships
│   │               ├── schema/
│   │               │   ├── base.py              # DocumentClassifier ABC; FieldExtractor ABC
│   │               │   ├── keyword_classifier.py # KeywordClassifier — zero-API-cost type detection from signal phrases
│   │               │   ├── llm_extractor.py     # LLMFieldExtractor — one GPT-4o-mini call/doc; type-specific fields
│   │               │   ├── structured_extractor.py # StructuredFieldExtractor — caller-supplied Pydantic/TypedDict/JSON Schema
│   │               │   └── schema_store.py      # RecordStore ABC; MemoryRecordStore; FileRecordStore; JsonSourceRecordStore
│   │               └── graph/
│   │                   └── graph_indexer.py     # GraphIndexer — writes Document/Section/Chunk/Entity nodes + edges during ingest
│   └── tests/
│       ├── unit/                            # 20 test files; all I/O mocked or in-memory
│       │   ├── test_basic_enricher.py
│       │   ├── test_document_find.py
│       │   ├── test_document_pipeline.py
│       │   ├── test_document_record.py
│       │   ├── test_document_source.py
│       │   ├── test_docx_parser.py
│       │   ├── test_graph_indexer.py
│       │   ├── test_hybrid_search.py
│       │   ├── test_llm_enricher.py
│       │   ├── test_nl_parser.py
│       │   ├── test_openai_embedding.py
│       │   ├── test_pdf_parser.py
│       │   ├── test_plumber_pdf_parser.py
│       │   ├── test_pre_filtering.py
│       │   ├── test_rich_search.py
│       │   ├── test_section_chunker.py
│       │   ├── test_sentence_chunker.py
│       │   ├── test_structured_extractor.py
│       │   ├── test_text_parser.py
│       │   └── test_window_chunker.py
│       ├── integration/                     # requires Redis Stack on localhost:6379
│       │   ├── conftest.py
│       │   └── test_end_to_end.py           # full ingest → search round-trip with RedisVectorSource
│       └── benchmarks/
│           ├── README.md
│           ├── RETRIEVAL_RESULTS.md         # recorded baseline retrieval benchmark results
│           ├── bench_chunkers.py            # SectionChunker vs WindowChunker throughput
│           ├── bench_pipeline.py            # full pipeline throughput (pages/sec)
│           ├── bench_retrieval.py           # retrieval quality: P@k and MRR on curated queries
│           ├── datasets.py                  # fixture documents and ground-truth query sets
│           └── runner.py                    # benchmark harness
│
└── platform/                                # external interface services
    ├── README.md                            # combined platform docs (API, CLI, MCP)
    ├── document-mcp/                        # MCP server (fastmcp)
    │   ├── README.md                        # package overview; defers to platform/README.md
    │   ├── src/
    │   │   └── server.py                    # fastmcp server; 7 MCP tools; DocumentPipeline singleton
    │   └── tests/
    │       └── test_server.py
    ├── api/                                 # REST API (FastAPI, port 8001)
    │   ├── README.md                        # package overview; defers to platform/README.md
    │   ├── pyproject.toml
    │   ├── src/doc_api/
    │   │   ├── main.py                      # FastAPI app; lifespan; uvicorn entry point (port 8001)
    │   │   ├── config.py                    # DocumentPipeline singleton from env vars
    │   │   └── routers/
    │   │       └── documents.py             # /documents/* routes; Pydantic request models
    └── cli/                                 # CLI (blue-data-doc-cli PyPI package)
        ├── README.md                        # package overview; defers to platform/README.md
        ├── pyproject.toml
        ├── src/blue_data_doc_cli/
        │   ├── main.py                      # Click command group: ensure-index, ingest, search, delete
        │   └── config.py                    # DocumentPipeline singleton from env vars
        └── tests/
            └── test_cli.py
```

---

## Layout Rationale

### `lib/` — the Python library

All core business logic lives under `lib/src/`. The `src/` layout prevents the package from being accidentally importable from the project root. `lib` is the only artifact published to PyPI as `blue-data-doc-source`.

### Pipeline stages map to subdirectories

The pipeline is a strict one-way chain. Each subdirectory owns exactly one stage:

```
parsers/  →  schema/  →  chunkers/  →  enrichment/  →  embeddings/  →  chunk_record  →  graph/
```

| Directory | Stage | Responsibility |
|-----------|-------|---------------|
| `parsers/` | Parse | Convert file bytes to page dicts; handle text, PDF, DOCX |
| `schema/` | Classify + Extract | Detect document type; extract typed fields; persist DocumentRecord |
| `chunkers/` | Chunk | Split extracted text into (section_title, text) pairs |
| `enrichment/` | Enrich | Annotate each chunk with entities and semantic_hints |
| `embeddings/` | Embed | Convert chunk text to a fixed-dimension float vector |
| `chunk_record.py` | Record | Assemble all provenance fields into `ChunkRecord` |
| `graph/` | Graph index | Write Document / Section / Chunk / Entity nodes and edges |

`document_source.py` and `pipeline.py` are the two coordination modules that sit above all stages. `pipeline.py` runs the chain; `document_source.py` dispatches on `op` codes and calls the pipeline.

### Extension points

Every stage exposes an ABC that callers can substitute:

| ABC | Concrete implementations |
|-----|------------------------|
| `Parser` | `TextParser`, `PDFParser`, `PlumberPDFParser`, `DocxParser`; or any custom subclass |
| `Chunker` | `SectionChunker` (default), `WindowChunker`, `SentenceChunker` |
| `EmbeddingModel` | `OpenAIEmbeddingModel`, `DummyEmbeddingModel` |
| `Enricher` | `BasicEnricher`, `LLMEnricher` |
| `DocumentClassifier` | `KeywordClassifier` |
| `FieldExtractor` | `LLMFieldExtractor`, `StructuredFieldExtractor` |
| `RecordStore` | `MemoryRecordStore`, `FileRecordStore`, `JsonSourceRecordStore` |
| `VectorSource` | Any blue-data `VectorSource` implementation |
| `GraphSource` | Any blue-data `GraphSource` implementation |

### `platform/` — external interface services

Three thin adapter services delegate all logic to a `DocumentPipeline` singleton. None contain business logic.

| Service | Package | Port / entry point |
|---------|---------|-------------------|
| `document-mcp/` | — | fastmcp stdio |
| `api/` | `blue-data-doc-api` | Uvicorn, port 8001 |
| `cli/` | `blue-data-doc-cli` | `blue-data-doc` Click command |
