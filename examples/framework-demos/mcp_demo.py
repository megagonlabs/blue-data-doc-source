"""
MCP Server Demo  (requires OPENAI_API_KEY + langchain-mcp-adapters)
====================================================================
Shows how the document-mcp FastMCP server exposes DocumentDataSource
as MCP tools that an LLM agent can call.

MCP tools exposed by platform/document-mcp/src/server.py:
  • ingest_document(path, collection)         → list of chunk records
  • search_documents(query, collection, k)    → list of chunk records
  • delete_document(path, collection)         → deletion status

This demo shows two things:
  1. Direct tool invocation via langchain-mcp-adapters + an LLM agent
  2. How the MCP server is configured (env vars, startup)

Requirements:
    pip install langchain-mcp-adapters langchain-openai langgraph
    OPENAI_API_KEY must be set
    The MCP server must be running (see "Start MCP server" below)

Start MCP server (in a separate terminal):
    cd platform/document-mcp
    BLUE_DOC_EMBEDDING=dummy \\
    BLUE_DOC_REDIS_HOST=localhost \\
    BLUE_DOC_REDIS_PORT=6379 \\
    python src/server.py

Run demo:
    OPENAI_API_KEY=sk-... python3 examples/framework-demos/mcp_demo.py
    OPENAI_API_KEY=sk-... python3 examples/framework-demos/mcp_demo.py --server-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

API_KEY = os.getenv("OPENAI_API_KEY", "")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY is not set.  Export it and retry.")
    sys.exit(1)

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
    from langchain_openai import ChatOpenAI  # type: ignore
    from langgraph.prebuilt import create_react_agent  # type: ignore
except ImportError:
    print("ERROR: langchain-mcp-adapters, langchain-openai, or langgraph is not installed.")
    print("  pip install langchain-mcp-adapters langchain-openai langgraph")
    sys.exit(1)

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_DOC = DATA_DIR / "sample_policy.txt"
DIV  = "=" * 68
DIV2 = "-" * 68

AGENT_QUESTIONS = [
    f"Please ingest the document at {DEFAULT_DOC} into the 'mcp_demo' collection, then search for 'deductible'.",
    "What does the policy say about prescription drug coverage?",
    "Search for emergency room information and summarize the copay structure.",
]


async def run_async(doc_path: str, server_url: str) -> None:
    print()
    print(DIV)
    print("MCP Server Demo")
    print(DIV)
    print()
    print(f"  MCP server : {server_url}")
    print(f"  Document   : {doc_path}")
    print()

    print(DIV2)
    print("MCP tool overview  —  what the server exposes")
    print(DIV2)
    print()
    print("  Tool: ingest_document(path, collection='default')")
    print("    → Ingests a file, returns chunk metadata list")
    print()
    print("  Tool: search_documents(query, collection='default', k=5)")
    print("    → KNN vector search, returns list of ChunkRecord dicts")
    print()
    print("  Tool: delete_document(path, collection='default')")
    print("    → Deletes all chunks for the given path")
    print()

    print(DIV2)
    print("Connecting to MCP server and creating agent …")
    print(DIV2)
    print()

    try:
        async with MultiServerMCPClient(
            {
                "document": {
                    "url": f"{server_url}/mcp",
                    "transport": "streamable_http",
                }
            }
        ) as client:
            tools = client.get_tools()
            tool_names = [t.name for t in tools]
            print(f"  Tools available: {tool_names}")
            print()

            llm   = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)
            agent = create_react_agent(llm, tools)

            print(DIV2)
            print("Agent sessions")
            print(DIV2)

            for question in AGENT_QUESTIONS:
                print()
                print(f"  User: {question}")
                result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
                messages = result.get("messages", [])
                answer   = messages[-1].content if messages else "(no response)"
                print(f"  Agent: {answer[:500]}…" if len(answer) > 500 else f"  Agent: {answer}")
                print()

    except Exception as e:
        print(f"  ERROR connecting to MCP server: {e}")
        print()
        _show_manual_test_guidance(server_url, doc_path)


def _show_manual_test_guidance(server_url: str, doc_path: str) -> None:
    print(DIV2)
    print("Manual test  —  curl to verify the server is running")
    print(DIV2)
    print()
    print("  Start the server first:")
    print("    cd platform/document-mcp")
    print("    BLUE_DOC_EMBEDDING=dummy BLUE_DOC_REDIS_HOST=localhost python src/server.py")
    print()
    print("  Then in another terminal:")
    print(f"    curl {server_url}/mcp")
    print()
    print("  Or use the FastMCP inspector:")
    print(f"    npx @modelcontextprotocol/inspector {server_url}/mcp")
    print()
    print(DIV2)
    print("MCP server configuration reference")
    print(DIV2)
    print()
    print("  Environment variables (platform/document-mcp):")
    print("    BLUE_DOC_EMBEDDING    dummy | openai   (default: openai)")
    print("    BLUE_DOC_OPENAI_KEY   sk-...           (required if embedding=openai)")
    print("    BLUE_DOC_REDIS_HOST   localhost")
    print("    BLUE_DOC_REDIS_PORT   6379")
    print("    BLUE_DOC_CHUNK_SIZE   200              (words per chunk)")
    print("    BLUE_DOC_OVERLAP      30               (overlap words)")
    print("    BLUE_DOC_COLLECTION   default          (vector store collection)")
    print()
    print("  The server is a FastMCP app — every tool is an async function")
    print("  that wraps one source.execute_query() call.")
    print()


def run(doc_path: str, server_url: str) -> None:
    import asyncio
    asyncio.run(run_async(doc_path, server_url))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP server demo")
    parser.add_argument("path",         nargs="?", default=str(DEFAULT_DOC))
    parser.add_argument("--server-url", default="http://localhost:8000")
    args = parser.parse_args()
    run(args.path, args.server_url)
