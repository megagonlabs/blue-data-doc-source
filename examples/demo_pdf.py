"""End-to-end demo: PDF → parse → chunk → embed → Redis → search.

Usage:
    # Start Redis Stack first:
    #   docker run -p 6379:6379 redis/redis-stack-server:latest

    python examples/demo_pdf.py mydoc.pdf --query "eligibility requirements"
    python examples/demo_pdf.py mydoc.pdf --query "eligibility requirements" --embedding openai
"""
import argparse
import sys

from blue_doc_source.embeddings import DummyEmbeddingModel, OpenAIEmbeddingModel
from blue_doc_source.pipeline import DocumentPipeline

sys.path.insert(0, "examples")
from local_redis_store import LocalRedisVectorStore

EMBEDDING_DIMS = {
    "dummy": 8,
    "openai": 1536,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end PDF ingestion + search demo")
    parser.add_argument("path", help="Path to a .pdf file")
    parser.add_argument("--query", required=True, help="Search query to run after ingestion")
    parser.add_argument("--embedding", choices=["dummy", "openai"], default="dummy")
    parser.add_argument("--openai-model", default="text-embedding-3-small")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--collection", default="demo_docs")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    if args.embedding == "openai":
        embedding_model = OpenAIEmbeddingModel(model=args.openai_model)
    else:
        embedding_model = DummyEmbeddingModel(dim=EMBEDDING_DIMS["dummy"])

    print(f"Connecting to Redis at {args.redis_host}:{args.redis_port} ...")
    try:
        vector_store = LocalRedisVectorStore(host=args.redis_host, port=args.redis_port)
    except Exception as e:
        print(f"Could not connect to Redis: {e}")
        print("Start Redis Stack with: docker run -p 6379:6379 redis/redis-stack-server:latest")
        sys.exit(1)

    dim = EMBEDDING_DIMS[args.embedding]
    vector_store.create_index(
        collection=args.collection,
        index_name=args.collection,
        dim=dim,
    )

    pipeline = DocumentPipeline(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=embedding_model,
        vector_store=vector_store,
        collection=args.collection,
    )

    print(f"\nIngesting {args.path} ...")
    records = pipeline.ingest(args.path)
    print(f"Stored {len(records)} chunks in Redis (collection='{args.collection}')")

    print(f"\nSearching: \"{args.query}\"")
    results = pipeline.search(args.query, k=args.top_k)

    if not results:
        print("No results returned.")
    else:
        for i, r in enumerate(results, 1):
            page = f"p{r['page_start']}" if r.get("page_start") else ""
            score = float(r.get("score", 0))
            snippet = r.get("text", "")[:200].replace("\n", " ")
            print(f"\n[{i}] {r.get('file_name', '')} {page}  score={score:.4f}")
            print(f"    {snippet}...")
