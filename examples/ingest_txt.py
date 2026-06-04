import argparse

from blue_doc_source.embeddings import DummyEmbeddingModel
from blue_doc_source.pipeline import DocumentPipeline


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("path", help="Path to a .txt file")
    arg_parser.add_argument("--chunk-size", type=int, default=1000)
    arg_parser.add_argument("--overlap", type=int, default=150)
    arg_parser.add_argument("--with-embeddings", action="store_true")

    args = arg_parser.parse_args()

    embedding_model = DummyEmbeddingModel(dim=8) if args.with_embeddings else None

    pipeline = DocumentPipeline(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=embedding_model,
    )

    records = pipeline.ingest(args.path)

    print(f"Generated {len(records)} chunk records")
    for record in records[:3]:
        print(record)