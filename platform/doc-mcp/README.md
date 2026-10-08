# blue-data-doc-mcp

MCP server that exposes blue-data-doc-source operations as AI-callable tools for Claude Desktop and other MCP clients.

Full documentation — configuration, tool list, and Claude Desktop integration — is in [platform/README.md](../README.md).

## Quick start

```bash
pip install -e lib/ -e platform/doc-mcp/
python platform/doc-mcp/src/server.py
```

## Testing

```bash
pip install -e lib/ -e platform/doc-mcp/ pytest
pytest platform/doc-mcp/tests/ -v
```

Tests stub the MCP SDK and patch the pipeline — no Redis or OpenAI connection needed.
