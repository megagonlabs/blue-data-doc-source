#!/usr/bin/env python
"""Download third-party document fixtures that are not committed to the repo.

Per the Disclosures section of the README, this repository does not redistribute
actual third-party datasets — only links to their original sources. A couple of
examples ingest real-world documents (an arXiv paper and a public annual report);
those files are git-ignored and downloaded on demand by this script.

Run once before the examples that use them:

    python scripts/fetch_fixtures.py
    python examples/feature-demos/pdf_formats_demo.py

Examples that need a document which is not present skip that step with a pointer
back to this script, so fetching is only required to exercise those steps.

Note: these fixtures are pinned to specific public documents. If an upstream file
is moved or revised, update the URL below (and the README "Disclosures > Datasets"
table). The documents are third-party content, outside this project's control.

Keep the ``_FIXTURES`` table below in sync with the Disclosures > Datasets table
in the README.
"""
from __future__ import annotations

from pathlib import Path

import httpx

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# fixture filename -> upstream source URL
_FIXTURES = {
    # "Attention Is All You Need" (Vaswani et al., 2017) — arXiv:1706.03762
    "real_research_paper.pdf": "https://arxiv.org/pdf/1706.03762",
    # Berkshire Hathaway Inc. 2023 Annual Report
    "real_annual_report.pdf": "https://www.berkshirehathaway.com/2023ar/2023ar.pdf",
}

_USER_AGENT = "blue-data-doc-source-fixtures/0.1 (compatible; httpx)"


def fetch_all() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        timeout=60.0, follow_redirects=True, headers={"User-Agent": _USER_AGENT}
    ) as client:
        for name, url in _FIXTURES.items():
            dest = _DATA_DIR / name
            print(f"fetching {url} -> {dest}")
            response = client.get(url)
            response.raise_for_status()
            dest.write_bytes(response.content)
            print(f"  saved {len(response.content):,} bytes")


if __name__ == "__main__":
    fetch_all()
