# blue-data-doc-mcp

MCP server that exposes blue-data-doc-source operations as AI-callable tools for Claude Desktop and other MCP clients.

Full documentation — configuration, tool list, and Claude Desktop integration — is in [platform/README.md](../README.md).

## Quick start

```bash
pip install -e lib/ -e platform/document-mcp/
python platform/document-mcp/src/server.py
```
