# examples/ingest_txt.py

import argparse

from blue_doc_source.pipeline import DocumentPipeline


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("path", help="Path to a .txt file")
    arg_parser.add_argument("--chunk-size", type=int, default=1000)
    arg_parser.add_argument("--overlap", type=int, default=150)

    args = arg_parser.parse_args()

    pipeline = DocumentPipeline(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    records = pipeline.ingest(args.path)

    print(f"Generated {len(records)} chunk records")
    for record in records[:3]:
        print(record)

        