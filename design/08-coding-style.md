# blue-data-doc-source — Coding Style

## Tools

| Tool | Role | Config location |
|------|------|----------------|
| `ruff` | Linter + formatter (replaces Black, isort, flake8) | `pyproject.toml` → `[tool.ruff]` |
| `pyright` (strict) | Static type checker | `pyproject.toml` → `[tool.pyright]` |
| `pytest` | Test runner | `pyproject.toml` → `[tool.pytest.ini_options]` |

All three are declared as `dev` dependencies. CI fails on any lint, type, or test failure.

---

## Python Version and Annotations

- Minimum Python: **3.10**
- All files begin with `from __future__ import annotations` to enable PEP 604 union syntax (`X | Y`) on Python 3.10 and postponed evaluation of annotations in ABCs.

---

## Linting and Formatting

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

- `ruff format` is the formatter (Black-compatible, 100-character lines).
- `ruff check --fix` auto-corrects safe lint violations.
- Imports are sorted by `ruff` with `I` rules: stdlib → third-party → local, alphabetical within each group.

---

## Type Annotations

- **All** public functions and methods carry full type annotations (parameters + return type).
- `pyright --strict` must pass with zero errors.
- Use `X | None` (not `Optional[X]`) on Python 3.10+ with `from __future__ import annotations`.
- Use `list[X]` / `dict[K, V]` / `tuple[X, ...]` (not `List`, `Dict`, `Tuple`) from `__future__` imports.
- `Any` is allowed only in ABCs where the concrete type is genuinely unknown (e.g. parser page-dict values, filter operator values).

---

## Synchronous Pipeline

The core library is **synchronous**. `DocumentPipeline` and all its stages (`Parser`, `Chunker`, `Enricher`, `EmbeddingModel`) are synchronous. There is no `async`/`await` in the library layer.

The REST API (`platform/doc-api/`) uses FastAPI with async route handlers, but the handlers call synchronous pipeline methods inside a `run_in_executor` shim or directly (for fast operations). The MCP server uses synchronous tool handlers.

---

## Naming

| Kind | Convention | Example |
|------|-----------|---------|
| Modules | `snake_case` | `section_chunker.py` |
| Classes | `PascalCase` | `SectionChunker`, `ChunkRecord` |
| Functions / methods | `snake_case` | `chunk_with_titles`, `embed_text` |
| Constants | `UPPER_SNAKE_CASE` | `_INDEX_EXTRA_FIELDS`, `BLUE_DOC_COLLECTION` |
| Private helpers | `_leading_underscore` | `_build_filter_expr` |
| Type aliases | `PascalCase` | `PageDict = dict[str, Any]` |

Op code strings follow a `NAMESPACE.VERB` pattern in `UPPER_CASE`: `DOCUMENT.INGEST`, `DOCUMENT.SEARCH`, `DOCUMENT.ENSURE_INDEX`.

---

## Abstract Base Classes

- Every pipeline stage exposes an ABC in its `base.py`.
- ABCs use `abc.ABC` + `@abstractmethod`. No `__init__` in ABCs unless shared initialization is required.
- Concrete implementations live in separate files (e.g. `section_chunker.py`, `window_chunker.py`).
- ABCs are exported from the package `__init__.py`; concrete classes are also exported for direct instantiation.

---

## Docstrings

- Single-line docstring for any public method whose name does not make its contract obvious.
- No multi-line docstrings unless a subtlety is non-obvious from the signature (hidden constraint, invariant, side-effect).
- Do **not** document what the code does — document **why** or **when** if surprising.

```python
def embed_text(self, text: str) -> list[float]:
    """Return a fixed-dimension embedding vector. Length must equal self.dim."""
```

---

## Error Handling

- All domain errors subclass a repo-local base: `DocSourceError(Exception)`.
- Sub-classes are defined per subsystem:
  - `ParseError` — unreadable or unsupported file
  - `ChunkError` — chunker contract violation
  - `EmbedError` — dimension mismatch or API failure
  - `IndexError` — vector index not created before ingest
- Do **not** catch and re-raise `Exception` broadly. Let unexpected errors propagate.
- Only catch exceptions at system boundaries (op dispatch in `execute_query`, CLI handlers, REST route handlers) and convert to appropriate HTTP status codes or CLI exit codes.

---

## Imports

- Deferred imports for optional extras: `pytesseract` and `pdf2image` are imported inside the OCR branch of `PDFParser`, not at module top-level. `nltk` is imported inside `SentenceChunker.chunk_with_titles()`. This ensures the package can be imported without extras installed.
- `openai` and `redis` are hard dependencies — they are imported at module top-level.

---

## File Layout

Each module follows this order:

1. `from __future__ import annotations`
2. stdlib imports
3. third-party imports
4. local imports
5. module-level constants / type aliases
6. class or function definitions

---

## Testing Style

- Test file names: `test_{module_name}.py`.
- Test function names: `test_{what}_{condition}` (e.g. `test_chunk_id_is_deterministic`, `test_to_dict_omits_none_page_start`).
- One `assert` per logical claim (multiple asserts in a test are fine when they verify the same invariant).
- No `print` in tests; use `assert` with descriptive messages if the failure reason is not obvious.
- Never mock internal collaborators — use real in-process test doubles (`MemoryVectorSource`, `MemoryRecordStore`, `DummyEmbeddingModel`). Mock only external APIs (`openai.OpenAI`) via `unittest.mock.patch`.
