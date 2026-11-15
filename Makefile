.PHONY: help install lint format test run check

help:  ## Show commands
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Create venv and install backend + dev deps
	cd backend && python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
	pre-commit install || true

lint:  ## Ruff lint + format check
	cd backend && ruff check . && ruff format --check .

format:  ## Auto-fix lint and format
	cd backend && ruff check --fix . && ruff format .

test:  ## Run tests with coverage
	cd backend && pytest

run:  ## Start API with hot reload on :8000
	cd backend && uvicorn app.main:app --reload --port 8000

check: lint test  ## What CI runs
