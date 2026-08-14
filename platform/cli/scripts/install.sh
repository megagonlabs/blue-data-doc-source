#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$CLI_DIR")")"

pip install -e "$REPO_ROOT/../blue-data"
pip install -e "$REPO_ROOT"
pip install -e "$CLI_DIR"
