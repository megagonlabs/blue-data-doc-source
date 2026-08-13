"""
LangGraph RAG Agent  (requires OPENAI_API_KEY + langgraph)
===========================================================
Builds a minimal two-node LangGraph RAG pipeline that uses
DocumentDataSource as its retrieval backend.

Graph:
    [retrieve] → [generate] → END

  retrieve: calls source.execute_query DOCUMENT.SEARCH
  generate: calls OpenAI chat with retrieved chunks as context

Requirements:
    pip install langgraph langchain-openai
    OPENAI_API_KEY environment variable must be set.

Run:
    OPENAI_API_KEY=sk-... python3 examples/framework-demos/langgraph_rag_demo.py
    OPENAI_API_KEY=sk-... python3 examples/framework-demos/langgraph_rag_demo.py --k 5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

API_KEY = os.getenv("OPENAI_API_KEY", "")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY is not set.  Export it and retry.")
    sys.exit(1)

try:
    from langgraph.graph import StateGraph, END  # type: ignore
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore  # noqa: F401
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
except ImportError:
    print("ERROR: langgraph or langchain-openai is not installed.")
    print("  pip install langgraph langchain-openai")
    sys.exit(1)

from blue_data_doc_source import DocumentDataSource, OpenAIEmbeddingModel, NLQueryParser

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_DOC = DATA_DIR / "sample_policy.txt"
COLLECTION  = "langgraph_rag"
DIV  = "=" * 68
DIV2 = "-" * 68

QUESTIONS = [
    "What is the annual deductible for an individual?",
    "How long do I have to enroll dependents after a qualifying life event?",
    "What are the prescription drug copay tiers?",
]


class RAGState(TypedDict):
    question: str
    chunks: list[dict]
    answer: str


def build_rag_graph(
    source: DocumentDataSource,
    llm: ChatOpenAI,
    k: int,
) -> object:
    def retrieve(state: RAGState) -> RAGState:
        chunks = source.execute_query({
            "op": "DOCUMENT.SEARCH",
            "query": state["question"],
            "collection": COLLECTION,
            "k": k,
        })
        return {**state, "chunks": chunks}

    def generate(state: RAGState) -> RAGState:
        context = "\n\n---\n\n".join(
            f"[{r.get('section_title', 'unknown')}] {r.get('text', '')}"
            for r in state["chunks"]
        )
        messages = [
            SystemMessage(content=(
                "You are a helpful insurance policy assistant. "
                "Answer the question using only the provided context. "
                "If the context does not contain enough information, say so."
            )),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}"),
        ]
        response = llm.invoke(messages)
        return {**state, "answer": response.content}

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


def run(doc_path: str, k: int) -> None:
    print()
    print(DIV)
    print("LangGraph RAG Agent")
    print(DIV)
    print()
    print(f"  Document  : {doc_path}")
    print(f"  Embedding : text-embedding-3-small")
    print(f"  LLM       : gpt-4o-mini")
    print(f"  k         : {k} chunks per retrieval")
    print()

    store  = LocalMemoryVectorStore()
    source = DocumentDataSource(
        name="langgraph_rag",
        embedding_model=OpenAIEmbeddingModel(
            api_key=API_KEY,
            model="text-embedding-3-small",
            dim=1536,
        ),
        vector_source=store,
        chunk_size=200,
        overlap=30,
        default_collection=COLLECTION,
        nl_parser=NLQueryParser(),
    )
    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": COLLECTION})

    print("  Ingesting (calling OpenAI for embeddings) …")
    source.execute_query({"op": "DOCUMENT.INGEST", "path": doc_path, "collection": COLLECTION})
    print(f"  ✓ Index ready with {store.count(COLLECTION)} chunks")
    print()

    llm   = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)
    graph = build_rag_graph(source, llm, k)

    print(DIV2)
    print("RAG graph  →  [retrieve] → [generate] → END")
    print(DIV2)
    print()

    for question in QUESTIONS:
        print(f"  Q: {question}")
        state = graph.invoke({"question": question, "chunks": [], "answer": ""})
        chunks = state.get("chunks", [])
        answer = state.get("answer", "")

        print(f"  Retrieved {len(chunks)} chunk(s):")
        for r in chunks[:2]:
            section = r.get("section_title", "")
            snippet = r.get("text", "")[:100].replace("\n", " ")
            print(f"    [{section}] {snippet}…")
        print()
        print(f"  A: {answer}")
        print()

    print(DIV)
    print("Graph structure")
    print(DIV)
    print()
    print("  State schema (TypedDict):")
    print("    question : str        — input from user")
    print("    chunks   : list[dict] — set by retrieve node (ChunkRecord dicts)")
    print("    answer   : str        — set by generate node")
    print()
    print("  Integration with blue-data-doc-source:")
    print("    source.execute_query(DOCUMENT.SEARCH) → list[dict] (ChunkRecord)")
    print("    Each dict carries text + provenance (file_name, page, section_title)")
    print("    The generate node cites section_title in context — attribution built-in")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph RAG demo")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_DOC))
    parser.add_argument("--k",  type=int,  default=4)
    args = parser.parse_args()
    run(args.path, args.k)
