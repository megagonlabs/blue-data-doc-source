"""Quick ingestion demo for .txt files — no vector store required.

Usage:
    python examples/ingest_txt.py                          # uses sample policy
    python examples/ingest_txt.py path/to/doc.txt
    python examples/ingest_txt.py data/sample_policy.txt --embedding dummy --query "deductible"
"""
import argparse
from pathlib import Path

from blue_doc_source import DocumentPipeline, DummyEmbeddingModel, OpenAIEmbeddingModel

DEFAULT_PATH = Path(__file__).parent.parent / "data" / "sample_policy.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=str(DEFAULT_PATH), help="Path to a .txt file")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--embedding", choices=["none", "dummy", "openai"], default="none")
    parser.add_argument("--openai-model", default="text-embedding-3-small")
    parser.add_argument("--query", help="Optional search query to run after ingestion")
    args = parser.parse_args()

    if args.embedding == "openai":
        embedding_model = OpenAIEmbeddingModel(model=args.openai_model)
    elif args.embedding == "dummy":
        embedding_model = DummyEmbeddingModel(dim=8)
    else:
        embedding_model = None

    pipeline = DocumentPipeline(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=embedding_model,
    )

    print(f"Ingesting: {args.path}")
    records = pipeline.ingest(args.path)
    print(f"Generated {len(records)} chunks\n")

    for r in records:
        section = f"[{r['section_title']}] " if r.get("section_title") else ""
        print(f"  {r['chunk_id']}  {section}{r['text'][:80]}...")

    if args.query:
        if embedding_model is None:
            print("\n--query requires --embedding dummy or --embedding openai")
        else:
            print(f"\nSearching: \"{args.query}\"")
            print("(no vector store configured — attach a vector_source to enable search)")
