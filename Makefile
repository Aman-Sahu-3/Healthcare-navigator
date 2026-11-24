.PHONY: help install lint format test run check ingest sources search web web-install web-test

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
	cd backend && uvicorn app.main:app --reload --reload-dir app --port 8000

ingest:  ## Ingest PDFs from data/raw into ChromaDB
	cd backend && python -m app.ingestion

sources:  ## List documents stored in ChromaDB
	cd backend && python -m app.ingestion --list

search:  ## Retrieval test: make search Q="first-line treatment for hypertension"
	cd backend && python -m app.ingestion --search "$(Q)"

web-install:  ## Install frontend dependencies
	cd frontend && npm ci

web:  ## Start the React dev server on :5173 (proxies /api to :8000)
	cd frontend && npm run dev

web-test:  ## Typecheck + test the frontend
	cd frontend && npm run typecheck && npm test

check: lint test web-test  ## What CI runs
