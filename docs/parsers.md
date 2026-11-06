# Parsers

The parser layer extracts raw text from a source file before it enters the chunking pipeline. Parsers are independent of chunking, enrichment, and embedding — swapping a parser has no effect on any downstream stage.

All parsers implement the `Parser` ABC:

```python
class Parser(ABC):
    def parse(self, path: str) -> List[Dict[str, Any]]:
        ...
```

The return value is a list of dicts. Each dict must contain `"text": str`. Include `"page_num": int` (1-based) to propagate page provenance to `ChunkRecord.page_start` and `page_end`.

---

## Built-in parsers

### TextParser

Reads a UTF-8 `.txt` file and returns a single page dict with the full file text.

```python
from blue_data_doc_source import TextParser

text = TextParser().parse("report.txt")
# [{"text": "Full file content..."}]
```

No external dependencies.

---

### PDFParser

Reads a `.pdf` file page by page using pypdf. Returns one dict per page with `page_num` set.

```python
from blue_data_doc_source.source.document.parsers import PDFParser

pages = PDFParser().parse_pages("contract.pdf")
# [{"page_num": 1, "text": "..."}, {"page_num": 2, "text": "..."}, ...]
```

Pages with no embedded text layer (scanned documents, photographed pages) return an empty `"text"` by default.

#### OCR fallback

Pass `ocr_fallback=True` to run Tesseract OCR on any page with no text layer instead of returning empty text:

```python
parser = PDFParser(ocr_fallback=True)
pages = parser.parse_pages("scanned_contract.pdf")
```

OCR pages include additional fields:

```python
{"page_num": 3, "text": "extracted text...", "ocr": True, "ocr_confidence": 87.4}
```

`ocr_confidence` is the mean word-level confidence score from Tesseract (0–100), computed from `image_to_data()` output filtered to non-empty tokens. This propagates to `ChunkRecord.metadata["ocr_confidence"]` and can be used downstream to flag or filter low-quality extractions.

**OCR preprocessing:** Before OCR, each page image is preprocessed with Pillow: converted to grayscale, contrast enhanced 2×, then binarized at a 128 threshold. This improves Tesseract accuracy on low-contrast or faded scans.

**Install OCR dependencies:**

```bash
pip install -e "lib/.[ocr]"             # pytesseract + pdf2image
brew install tesseract                   # macOS — Tesseract must be on PATH
brew install poppler                     # macOS — required by pdf2image
```

On Ubuntu: `apt-get install tesseract-ocr poppler-utils`

A clear `ImportError` is raised if `ocr_fallback=True` is used without the dependencies installed.

---

### DocxParser

Reads a `.docx` file using python-docx, preserving heading hierarchy and table structure. Returns a single page dict with markdown-formatted text.

```python
from blue_data_doc_source import DocxParser

text = DocxParser().parse("policy.docx")
# [{"text": "# Introduction\n\nThis policy...\n\n## Eligibility\n\n| Month | Amount |\n..."}]
```

**Heading conversion:** Word heading styles map to markdown `#` prefixes:

| Word style | Output |
|---|---|
| Heading 1 / Title | `# text` |
| Heading 2 / Subtitle | `## text` |
| Heading 3 | `### text` |
| Heading 4–6 | `#### text` – `###### text` |

**Table rendering:** Tables are converted to pipe-delimited rows:

```
| Name | Amount |
| Alice | 100 |
| Bob | 200 |
```

**Document order:** The parser iterates `doc.element.body` XML directly (not `doc.paragraphs` and `doc.tables` separately) to preserve the interleaved order of paragraphs and tables as they appear in the file.

Because `DocxParser` outputs markdown-formatted text, `SectionChunker` detects the heading structure and assigns correct `section_title` values — Word documents with headings produce fully structured chunk records without any extra configuration.

---

### PlumberPDFParser

`PlumberPDFParser` is a block-aware PDF parser built on [pdfplumber](https://github.com/jsvine/pdfplumber). Unlike `PDFParser` (which extracts a flat text string per page), `PlumberPDFParser` separates tables from prose and returns typed blocks.

```python
from blue_data_doc_source import PlumberPDFParser

parser = PlumberPDFParser()
blocks = parser.parse_blocks("invoice.pdf")

for block in blocks:
    if block["type"] == "table":
        print("TABLE:", block["table"]["headers"])
        for row in block["table"]["rows"]:
            print(" ", row)
    else:
        print("TEXT:", block["text"][:80])
```

Each block in the returned list is one of:

| `type` | Fields present | Notes |
|---|---|---|
| `"text"` | `type`, `text`, `page_num` | Text outside any table region |
| `"table"` | `type`, `text`, `page_num`, `table` | `table = {"headers": list[str], "rows": list[list[str]]}` |

The `text` field on a table block is a pipe-delimited representation of the full table (header + rows), suitable for embedding and search without additional processing.

When used with `DocumentPipeline`, table blocks bypass the chunker — each table becomes exactly one `ChunkRecord` with `content_type="table"`. Text blocks are chunked normally. The `content_type` field is indexed as a filterable TAG, so you can filter search results to tables only:

```python
from blue_data_doc_source import DocumentPipeline, PlumberPDFParser

pipeline = DocumentPipeline(
    parser=PlumberPDFParser(),
    embedding_model=...,
    vector_source=...,
)
pipeline.ingest("invoice.pdf")

# Only return table chunks
results = pipeline.search("total amount", filters={"content_type": "table"})
```

`parse()` is also implemented for backward compatibility — it flattens all blocks into plain page dicts so `PlumberPDFParser` can be used anywhere `PDFParser` is accepted.

**Install:** `pip install pdfplumber`

---

## Using the pipeline with built-in parsers

`DocumentPipeline.ingest()` auto-detects the parser by file extension:

```python
pipeline.ingest("report.txt")      # TextParser
pipeline.ingest("contract.pdf")    # PDFParser
pipeline.ingest("policy.docx")     # DocxParser
```

Pass `ocr_fallback=True` at pipeline construction to enable OCR for PDFs:

```python
pipeline = DocumentPipeline(ocr_fallback=True, ...)
pipeline.ingest("scanned.pdf")
```

---

## Pluggable parser: bring your own

When the built-in parsers are insufficient — complex layouts, multi-column PDFs, vision-model output, cloud OCR services — implement the `Parser` ABC and inject it:

```python
from blue_data_doc_source import DocumentPipeline, Parser

class LlamaIndexParser(Parser):
    def parse(self, path: str):
        from llama_index.core import SimpleDirectoryReader
        docs = SimpleDirectoryReader(input_files=[path]).load_data()
        return [{"text": doc.text} for doc in docs]

pipeline = DocumentPipeline(parser=LlamaIndexParser(), embedding_model=...)
pipeline.ingest("complex_layout.pdf")
```

When `parser=` is provided, the built-in `TextParser`, `PDFParser`, and `DocxParser` are bypassed. All downstream stages (chunking, enrichment, embedding, storage) are unchanged.

If your parser produces page-level output, include `"page_num"`:

```python
return [
    {"page_num": 1, "text": "..."},
    {"page_num": 2, "text": "..."},
]
```

This propagates to `ChunkRecord.page_start` and `page_end`, enabling page-level citation by downstream agents.

---

## Bypassing file parsing entirely

When text has already been extracted by an external tool, skip the parser and feed content directly into the pipeline:

```python
# Pre-extracted full-document text
pipeline.ingest_text(text, file_name="report.txt")

# Pre-extracted page-level output from a vision model or cloud OCR
pipeline.ingest_pages(
    [
        {"page_num": 1, "text": "Cover page text..."},
        {"page_num": 2, "text": "Section one..."},
    ],
    file_name="scanned_report.pdf",
)
```

`document_id` is derived from a SHA-256 hash of the content, so calling these methods twice with identical content is safe — it overwrites rather than duplicates in the vector store.

---

## MIME types

Each parser sets the `mime_type` field on every `ChunkRecord` it produces:

| Parser | mime_type |
|---|---|
| `TextParser` | `text/plain` |
| `PDFParser` | `application/pdf` |
| `DocxParser` | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` |
| `PlumberPDFParser` | `application/pdf` |
| Custom parser (`.pdf`) | `application/pdf` |
| Custom parser (other) | `text/plain` |
