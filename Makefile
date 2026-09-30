# ---------------------------------------------------------------------------
# MCP for Copilot — developer tasks
# ---------------------------------------------------------------------------
.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format typecheck test test-unit test-integration test-e2e \
        coverage check build clean docs docs-serve docker-build docker-up docker-down \
        pre-commit release-dry

PYTHON ?= python
PACKAGE := mcp_for_copilot
SRC := src/$(PACKAGE)
TESTS := tests

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# -- setup ------------------------------------------------------------------
install: ## Install the package (library + server extras)
	$(PYTHON) -m pip install -e ".[all]"

install-dev: ## Install with all extras and dev tooling, then install git hooks
	$(PYTHON) -m pip install -e ".[all,dev]"
	pre-commit install

# -- quality ----------------------------------------------------------------
lint: ## Run ruff check and format check
	ruff check $(SRC) $(TESTS)
	ruff format --check $(SRC) $(TESTS)

format: ## Auto-fix lint issues and format the code
	ruff check --fix $(SRC) $(TESTS)
	ruff format $(SRC) $(TESTS)

typecheck: ## Run mypy in strict mode
	mypy $(SRC)

# -- tests ------------------------------------------------------------------
test: ## Run the full test suite with coverage
	pytest --cov=$(PACKAGE) --cov-report=term-missing --cov-report=xml

test-unit: ## Run unit tests only (fast)
	pytest $(TESTS)/unit -q

test-integration: ## Run integration tests only
	pytest $(TESTS)/integration -q

test-e2e: ## Run end-to-end tests only
	pytest $(TESTS)/e2e -q

coverage: ## Open the HTML coverage report
	pytest --cov=$(PACKAGE) --cov-report=html
	@echo "Open htmlcov/index.html"

check: lint typecheck test ## Run every check (lint + types + tests)

# -- packaging --------------------------------------------------------------
build: clean ## Build the sdist and wheel
	$(PYTHON) -m build
	@ls -la dist/

release-dry: ## Build and validate the distribution metadata
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache \
		.coverage coverage.xml htmlcov site
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# -- docs -------------------------------------------------------------------
docs: ## Build the documentation site
	mkdocs build --strict

docs-serve: ## Serve the docs locally with live reload
	mkdocs serve

# -- docker -----------------------------------------------------------------
docker-build: ## Build the Docker image
	docker build -t mcp-for-copilot:local .

docker-up: ## Start the local stack
	docker compose up -d

docker-down: ## Stop the local stack
	docker compose down

# -- hooks ------------------------------------------------------------------
pre-commit: ## Run all pre-commit hooks against every file
	pre-commit run --all-files
