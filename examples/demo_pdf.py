"""End-to-end demo: PDF -> parse -> chunk -> embed -> Redis -> search.

Usage:
    # Start Redis Stack first:
    #   docker run -p 6379:6379 redis/redis-stack-server:latest

    python examples/demo_pdf.py                            # uses sample policy
    python examples/demo_pdf.py data/sample_policy.pdf --query "what is the deductible"
    python examples/demo_pdf.py mydoc.pdf --query "eligibility" --embedding openai
"""
import argparse
import sys
from pathlib import Path

from blue_doc_source import DocumentDataSource, DummyEmbeddingModel, OpenAIEmbeddingModel

sys.path.insert(0, str(Path(__file__).parent))
from local_redis_store import LocalRedisVectorStore

DEFAULT_PATH = Path(__file__).parent.parent / "data" / "sample_policy.pdf"
DEFAULT_QUERY = "what is the deductible and out-of-pocket maximum"

EMBEDDING_DIMS = {
    "dummy": 8,
    "openai": 1536,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end PDF ingestion + search demo")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_PATH), help="Path to a .pdf file")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Search query to run after ingestion")
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

    source = DocumentDataSource(
        name="demo",
        embedding_model=embedding_model,
        vector_source=vector_store,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        default_collection=args.collection,
    )

    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": args.collection})

    print(f"Ingesting: {args.path}")
    records = source.execute_query({
        "op": "DOCUMENT.INGEST",
        "path": args.path,
        "collection": args.collection,
    })
    print(f"Stored {len(records)} chunks (collection='{args.collection}')\n")

    print(f"Searching: \"{args.query}\"")
    results = source.execute_query({
        "op": "DOCUMENT.SEARCH",
        "query": args.query,
        "collection": args.collection,
        "k": args.top_k,
    })

    if not results:
        print("No results returned.")
    else:
        for i, r in enumerate(results, 1):
            page = f"p{r['page_start']}" if r.get("page_start") else ""
            section = f"  [{r['section_title']}]" if r.get("section_title") else ""
            score = float(r.get("score", 0))
            snippet = r.get("text", "")[:200].replace("\n", " ")
            print(f"[{i}] {r.get('file_name', '')} {page}  score={score:.4f}{section}")
            print(f"    {snippet}...\n")
