#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"          # lib/
REPO_ROOT="$(dirname "$PACKAGE_DIR")"           # blue-data-doc-source/
PROJECTS_DIR="$(dirname "$REPO_ROOT")"          # parent dir containing sibling repos

pip install -e "$PROJECTS_DIR/blue-data/lib"
pip install -e "$PACKAGE_DIR"
