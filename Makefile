.PHONY: help sync lint format typecheck test cov check clean

help:
	@echo "make sync       - install dependencies with uv"
	@echo "make lint       - ruff lint"
	@echo "make format     - apply ruff formatting"
	@echo "make typecheck  - run mypy"
	@echo "make test       - run pytest"
	@echo "make cov        - run pytest with coverage"
	@echo "make check      - lint + format check + typecheck + test"

sync:
	uv sync --group dev

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

cov:
	uv run pytest --cov=chunkasr --cov-report=term-missing

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest

clean:
	rm -rf build dist .ruff_cache .mypy_cache .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
