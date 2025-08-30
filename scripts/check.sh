#!/usr/bin/env bash
# Run the full local gate, mirroring CI.
set -euo pipefail

echo "==> ruff check"
uv run ruff check .

echo "==> ruff format --check"
uv run ruff format --check .

echo "==> mypy"
uv run mypy

echo "==> pytest"
uv run pytest "$@"

echo "all checks passed"
