import argparse

from blue_doc_source.source.document.embeddings import DummyEmbeddingModel, OpenAIEmbeddingModel
from blue_doc_source.source.document.pipeline import DocumentPipeline


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("path", help="Path to a .txt file")
    arg_parser.add_argument("--chunk-size", type=int, default=1000)
    arg_parser.add_argument("--overlap", type=int, default=150)
    arg_parser.add_argument("--embedding", choices=["none", "dummy", "openai"], default="none")
    arg_parser.add_argument("--openai-model", default="text-embedding-3-small")
    arg_parser.add_argument("--query", help="Optional search query to run after ingestion")

    args = arg_parser.parse_args()

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

    records = pipeline.ingest(args.path)
    print(f"Generated {len(records)} chunk records")
    for record in records[:3]:
        print(record)

    if args.query:
        print(f"\nSearching for: {args.query}")
        results = pipeline.search(args.query)
        for r in results:
            print(r)
