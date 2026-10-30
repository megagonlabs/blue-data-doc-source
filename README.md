# blue-data-doc-source

Turn PDFs, DOCX, and text files into a queryable, filterable blue-data source.

Documents are data — not merely search-index content. `blue-data-doc-source` handles the full path from raw file to queryable vector index, and plugs into blue-data's `DataRegistry` so document evidence becomes a first-class member of the same intelligence layer as your SQL tables, graph stores, and web content.

→ **[Full documentation](docs/README.md)** · **[Quick start](docs/quickstart.md)** · **[Installation](docs/installation.md)** · **[Examples](examples/README.md)** · **[CLI](docs/cli.md)** · **[MCP server](docs/mcp.md)**

---

## What it does

- **Ingest** PDF, DOCX, and text files — parse, classify, chunk, enrich, embed, and upsert in one call.
- **Classify** every document by type at zero API cost and propagate it to every chunk for pre-filtered search.
- **Extract** typed fields at ingest (parties, totals, dates) via an LLM or a caller-supplied schema, then filter on them deterministically with `DOCUMENT.FIND` — no embedding, no query-time LLM.
- **Search** by semantic similarity with metadata pre-filtering, or combine BM25 + vector in one op.
- **Navigate** an optional entity/structure graph of documents, sections, chunks, and entities for exact entity lookup (structural + mention/co-occurrence edges; not a semantic relationship graph — see [docs/features.md](docs/features.md)).
- **Register** documents in blue-data's `DataRegistry` alongside SQL, web, and graph sources, with a consistent chunk schema.

Full feature list: [docs/features.md](docs/features.md). Runnable demos: [examples/](examples/).

---

## Repository layout

```
blue-data-doc-source/
├── lib/                    # Core Python library (blue-data-doc-source)
│   ├── src/blue_data_doc_source/
│   └── tests/              # unit, integration, and benchmark tests
├── platform/
│   ├── doc-cli/            # `blue-docdb` command-line interface
│   ├── doc-mcp/            # MCP server (Claude Desktop / Claude Code)
│   └── doc-api/            # FastAPI REST service (port 8001)
├── docs/                   # Documentation
├── examples/               # Runnable example scripts with captured output
├── design/                 # Design notes and decision log
└── data/                   # Sample documents used by the examples
```

---

## Requirements

- Python 3.12 or 3.13
- Sibling repo cloned beside this one: [`blue-data`](../blue-data)
- `OPENAI_API_KEY` — for OpenAI embeddings, LLM field extraction/enrichment, and natural-language queries (not needed for parsing, classification, or dummy embeddings):

```bash
export OPENAI_API_KEY=sk-...
```

Optional extras (`[ocr]`, `[nlp]`, `[mcp]`) are documented in [docs/installation.md](docs/installation.md).

---

## Quick start

```bash
# Clone blue-data beside this repo first, then:
python -m venv .venv && source .venv/bin/activate
bash lib/scripts/install.sh          # installs blue-data and lib/

# Ingest a document and inspect the extracted chunks — no API key required:
python - <<'PY'
from blue_data_doc_source import DocumentPipeline

for r in DocumentPipeline(chunk_size=500, overlap=100).ingest("data/sample_policy.txt"):
    print(r["chunk_id"], r.get("page_start"), r["text"][:80])
PY
```

For the full end-to-end walkthrough — embeddings, a vector store, graph indexing, and search — see [docs/quickstart.md](docs/quickstart.md) and the [guide](docs/guide.md).

---

## Packages

| Package | Description |
|---|---|
| [`lib/`](lib/README.md) | Core library `blue-data-doc-source` — `DocumentDataSource`, `DocumentPipeline`, and all ingestion components. Install: `bash lib/scripts/install.sh` |
| [`platform/doc-cli/`](platform/doc-cli/README.md) | `blue-docdb` command-line interface over the core ops. Install: `pip install -e platform/doc-cli` |
| [`platform/doc-mcp/`](platform/doc-mcp/README.md) | MCP server exposing document ops as AI-callable tools. Install: `pip install -e platform/doc-mcp` |
| [`platform/doc-api/`](platform/doc-api/README.md) | FastAPI REST service on port 8001. Install: `pip install -e platform/doc-api` |

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Report unacceptable behavior
to contact_oss@megagon.ai.

---

## Disclosures

This software may include, incorporate, or access open source software (OSS) components,
datasets and other third party components, including those identified below. The license terms
respectively governing the datasets and third-party components continue to govern those
portions, and you agree to those license terms may limit any distribution, use, and copying.
You may use any OSS components under the terms of their respective licenses, which may
include BSD 3, Apache 2.0, and other licenses. In the event of conflicts between Megagon Labs,
Inc. (“Megagon”) license conditions and the OSS license conditions, the applicable OSS
conditions governing the corresponding OSS components shall prevail.
You agree not to, and are not permitted to, distribute actual datasets used with the OSS
components listed below. You agree and are limited to distribute only links to datasets from
known sources by listing them in the datasets overview table below. You agree that any right to
modify datasets originating from parties other than Megagon are governed by the respective
third party’s license conditions.
You agree that Megagon grants no license as to any of its intellectual property and patent rights.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS (INCLUDING
MEGAGON) “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. You agree to cease using,
incorporating, and distributing any part of the provided materials if you do not agree with the
terms or the lack of any warranty herein.
While Megagon makes commercially reasonable efforts to ensure that citations in this
document are complete and accurate, errors may occur. If you see any error or omission, please
help us improve this document by sending information to contact_oss@megagon.ai.

### Datasets

All datasets used within the product are listed below (including their copyright holders and the license information).

For Datasets having different portions released under different licenses, please refer to the included source link specified for each of the respective datasets for identifications of dataset files released under the identified licenses.

</br>

| ID  | Dataset Name | Modified | Copyright Holder | Upstream Link | License  |
|-----|----------------------------------|----------|------------------|-----------------------------------------------------------------------------------------------------------|--------------------|
| 1 | "Attention Is All You Need" (example document `data/real_research_paper.pdf`) | No | Ashish Vaswani et al. (Google) | [link](https://arxiv.org/abs/1706.03762) | © the authors; distributed under the [arXiv.org non-exclusive license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| 2 | Berkshire Hathaway Inc. 2023 Annual Report (example document `data/real_annual_report.pdf`) | No | Berkshire Hathaway Inc. | [link](https://www.berkshirehathaway.com/2023ar/2023ar.pdf) | © Berkshire Hathaway Inc. (all rights reserved) |

### Open Source Software (OSS) Components

All OSS components used within the product are listed below (including their copyright holders and the license information).

For OSS components having different portions released under different licenses, please refer to the included Upstream link(s) specified for each of the respective OSS components for identifications of code files released under the identified licenses.

</br>

| ID  | OSS Component Name | Modified | Copyright Holder | Upstream Link | License  |
|-----|----------------------------------|----------|------------------|-----------------------------------------------------------------------------------------------------------|--------------------|
| 1 | click | No | Pallets | [link](https://github.com/pallets/click) | BSD License |
| 2 | fastapi | No | Sebastián Ramírez | [link](https://github.com/fastapi/fastapi) | MIT License |
| 3 | MCP | No | Anthropic, PBC | [link](https://github.com/modelcontextprotocol/python-sdk) | MIT License |
| 4 | nltk | No | NLTK Project | [link](https://github.com/nltk/nltk) | Apache Software License |
| 5 | openai | No | OpenAI | [link](https://github.com/openai/openai-python) | Apache Software License |
| 6 | pdf2image | No | Edouard Belval | [link](https://github.com/Belval/pdf2image) | MIT License |
| 7 | pydantic | No | Pydantic Services Inc. and individual contributors | [link](https://github.com/pydantic/pydantic) | MIT License |
| 8 | pypdf | No | Mathieu Fenniak and pypdf contributors | [link](https://github.com/py-pdf/pypdf) | BSD License |
| 9 | pytesseract | No | Samuel Hoffstaetter | [link](https://github.com/madmaze/pytesseract) | Apache Software License |
| 10 | python-docx | No | Steve Canny | [link](https://github.com/python-openxml/python-docx) | MIT License |
| 11 | redis | No | Redis, inc. | [link](https://github.com/redis/redis-py) | MIT License |
| 12 | rich | No | Will McGugan | [link](https://github.com/Textualize/rich) | MIT License |
| 13 | uvicorn | No | Encode OSS Ltd | [link](https://github.com/encode/uvicorn) | BSD License |
