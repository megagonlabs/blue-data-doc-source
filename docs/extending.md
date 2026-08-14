# Extending blue-data-doc-source

The pipeline has five extension points, each backed by an ABC:

| Extension point | ABC | Covered in |
|---|---|---|
| Custom parser | `Parser` | [parsers.md](parsers.md) |
| Custom chunker | `Chunker` | [chunkers.md](chunkers.md) |
| Custom embedding model | `EmbeddingModel` | This doc |
| Custom enricher | `Enricher` | This doc |
| Custom NL query parser | `NLQueryParser.parse_fn` | This doc |

---

## Custom EmbeddingModel

Implement `EmbeddingModel` when you want to use a local model, a non-OpenAI provider, or a batched embedding service.

```python
from blue_data_doc_source import EmbeddingModel
from typing import List

class MyEmbeddingModel(EmbeddingModel):
    @property
    def dim(self) -> int:
        return 768  # must match your model's output dimension

    def embed_text(self, text: str) -> List[float]:
        return my_model.encode(text).tolist()  # must return List[float]
```

Inject it anywhere an `EmbeddingModel` is accepted:

```python
from blue_data_doc_source import DocumentPipeline

pipeline = DocumentPipeline(
    embedding_model=MyEmbeddingModel(),
    vector_source=my_vector_source,
)
```

**`dim` contract** — the pipeline calls `ensure_index` with `dim=embedding_model.dim` when creating a vector index, and validates that every embedding returned by `embed_text` has exactly `dim` elements. A mismatch raises `ValueError`. Make sure `dim` reflects the actual output size of your model.

**Example: sentence-transformers**

```python
from sentence_transformers import SentenceTransformer
from blue_data_doc_source import EmbeddingModel
from typing import List

class SentenceTransformerModel(EmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()
```

---

## Custom Enricher

Implement `Enricher` when you want ML-based NER, domain-specific entity extraction, or metadata signals beyond what `BasicEnricher`'s regex patterns provide.

```python
from blue_data_doc_source import Enricher
from typing import Any, Dict

class MyEnricher(Enricher):
    def enrich(self, text: str) -> Dict[str, Any]:
        return {
            "entities": [
                {"text": "...", "type": "CLAUSE_REFERENCE"},
            ],
            "semantic_hints": {
                "word_count": len(text.split()),
                "is_liability_clause": "liability" in text.lower(),
            },
        }
```

The return dict must have exactly two keys:

| Key | Type | Description |
|---|---|---|
| `entities` | `list[dict]` | Each dict must have `"text": str` and `"type": str` |
| `semantic_hints` | `dict` | Any scalar metadata; propagates to `ChunkRecord.semantic_hints` |

Inject it at pipeline or source construction:

```python
pipeline = DocumentPipeline(enricher=MyEnricher(), ...)
source = DocumentDataSource(enricher=MyEnricher(), ...)
```

**Example: spaCy NER**

```python
import spacy
from blue_data_doc_source import Enricher
from typing import Any, Dict

class SpacyEnricher(Enricher):
    def __init__(self, model: str = "en_core_web_sm"):
        self._nlp = spacy.load(model)

    def enrich(self, text: str) -> Dict[str, Any]:
        doc = self._nlp(text)
        entities = [{"text": ent.text, "type": ent.label_} for ent in doc.ents]
        return {
            "entities": entities,
            "semantic_hints": {
                "word_count": len(text.split()),
                "sentence_count": len(list(doc.sents)),
                "keywords": [t.lemma_ for t in doc if t.is_alpha and not t.is_stop][:10],
            },
        }
```

**Combining enrichers** — the `Enricher` ABC does not provide a combiner, but it's straightforward to compose two enrichers:

```python
class CombinedEnricher(Enricher):
    def __init__(self, *enrichers: Enricher):
        self._enrichers = enrichers

    def enrich(self, text: str) -> Dict[str, Any]:
        all_entities = []
        merged_hints: Dict[str, Any] = {}
        for e in self._enrichers:
            result = e.enrich(text)
            all_entities.extend(result.get("entities", []))
            merged_hints.update(result.get("semantic_hints", {}))
        return {"entities": all_entities, "semantic_hints": merged_hints}

pipeline = DocumentPipeline(
    enricher=CombinedEnricher(BasicEnricher(), SpacyEnricher()),
)
```

---

## Custom NL query parser

`NLQueryParser` accepts a `parse_fn` callable — `(text: str) -> dict` — that maps a plain-English instruction to a `DOCUMENT.*` op dict. Use this to plug in any LLM backend instead of the default OpenAI call.

```python
from blue_data_doc_source import NLQueryParser, DocumentDataSource

def my_parse_fn(text: str) -> dict:
    # Call your own LLM and return a DOCUMENT.* op dict
    response = my_llm_call(text)
    return parse_response(response)

source = DocumentDataSource(
    name="contracts",
    embedding_model=my_embedding_model,
    vector_source=my_vector_source,
    nl_parser=NLQueryParser(parse_fn=my_parse_fn),
)

results = source.execute_query_nl("find all payment clauses")
```

The returned dict must be a valid op dict that `execute_query` can dispatch:

```python
# Valid op dicts
{"op": "DOCUMENT.SEARCH",  "query": "payment clause", "collection": "contracts", "k": 5}
{"op": "DOCUMENT.INGEST",  "path": "/docs/policy.pdf", "collection": "contracts"}
{"op": "DOCUMENT.DELETE",  "document_id": "doc_abc123", "chunk_count": 14}
{"op": "DOCUMENT.ENSURE_INDEX", "collection": "contracts"}
```

---

## Custom parsers and chunkers

See [parsers.md](parsers.md) for implementing a custom `Parser` and bypassing file parsing entirely via `ingest_text` / `ingest_pages`.

See [chunkers.md](chunkers.md) for implementing a custom `Chunker` and tuning `chunk_size` / `overlap`.
