.PHONY: help install install-tools sync test lint check-compat security format clean all pre-commit-install pre-commit-run generate-licenses check-licenses

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install project dependencies
	uv sync

install-tools: ## Install development tools
	uv tool install ruff
	uv tool install vermin
	uv tool install pre-commit --with pre-commit-uv
	uv tool install licensecheck

sync: install ## Alias for install

setup: install install-tools ## Complete development setup (install deps + tools)

# Development targets
test: ## Run tests
	uv run pytest

lint: ## Run linting checks
	uv tool run ruff check .

format: ## Format code with ruff
	uv tool run ruff check --fix .
	uv tool run ruff format .

check-format: ## Check if code is formatted correctly
	uv tool run ruff format --check .

check-compat: ## Check Python 3.11 compatibility
	uv tool run vermin --target=3.11- --no-tips --violations .

security: ## Check for security vulnerabilities
	uv run pip-audit .

generate-licenses: ## Generate licenses
	uv tool run licensecheck --license MIT \
		--format markdown --file THIRD_PARTY_NOTICES.md

check-licenses: ## Check for licenses
	uv tool run licensecheck --license MIT --show-only-failing --zero

# Pre-commit targets
pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

# Combined targets
check: lint check-format check-compat security check-licenses ## Run all checks (lint, format, compatibility, security)

all: format check test ## Format, check, and test

# CI simulation
ci: check test ## Run the same checks as CI

clean: ## Clean up build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/