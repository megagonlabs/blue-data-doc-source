# blue-data-doc-cli

Command-line interface for blue-data-doc-source. Ingest, search, and delete documents from a terminal.

Full documentation — configuration, commands, and examples — is in [platform/README.md](../README.md).

## Quick start

```bash
pip install -e lib/ -e platform/doc-cli/
doc ingest mydoc.txt
doc search "termination clause"
```

## Testing

```bash
pip install -e lib/ -e platform/doc-cli/ pytest
pytest platform/doc-cli/tests/ -v
```
