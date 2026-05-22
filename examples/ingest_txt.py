import argparse
from pathlib import Path
from uuid import uuid4

from blue_doc_source.chunkers import WindowChunker
from blue_doc_source.models import ChunkRecord
from blue_doc_source.parsers import TextParser


def ingest_txt(path: str, chunk_size: int = 1000, overlap: int = 150):
    parser = TextParser()
    chunker = WindowChunker(chunk_size=chunk_size, overlap=overlap)

    text = parser.parse(path)
    chunks = chunker.chunk(text)

    file_path = Path(path)
    document_id = f"doc_{uuid4().hex}"

    records = []
    for idx, chunk_text in enumerate(chunks):
        chunk_id = f"{document_id}_chunk_{idx:04d}"

        record = ChunkRecord(
            document_id=document_id,
            chunk_id=chunk_id,
            file_name=file_path.name,
            text=chunk_text,
            metadata={
                "source_path": str(file_path),
                "chunk_index": idx,
                "chunk_size": chunk_size,
                "overlap": overlap,
            },
        )

        records.append(record.to_dict())

    return records


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("path", help="Path to a .txt file")
    arg_parser.add_argument("--chunk-size", type=int, default=1000)
    arg_parser.add_argument("--overlap", type=int, default=150)

    args = arg_parser.parse_args()

    chunk_records = ingest_txt(
        args.path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print(f"Generated {len(chunk_records)} chunk records")
    for record in chunk_records[:3]:
        print(record)