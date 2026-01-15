.PHONY: help install install-tools sync test lint check-compat format clean all pre-commit-install pre-commit-run

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
	uv tool install basedpyright
	uv tool install pre-commit --with pre-commit-uv

sync: install ## Alias for install

setup: install install-tools pre-commit-install ## Complete development setup (install deps + tools)

pre-commit-install: ## Setup pre-commit hooks
	uv tool run pre-commit install

# Development targets
test: ## Run all tests (with generator if available)
	uv run pytest

test-package-conflict: ## Test package conflict
	@echo "Testing package conflict..."
	@echo "Creating virtual environment..."
	uv venv --seed -p 3.11 .venv-package-conflict
	@echo "Installing giskard..."
	.venv-package-conflict/bin/pip install giskard
	@echo "Installing giskard-core..."
	.venv-package-conflict/bin/pip install .
	@echo "Testing import giskard.core raises expected error..."
	@ERROR_OUTPUT=$$(.venv-package-conflict/bin/python -c "import giskard.core" 2>&1) || true; \
	echo "$$ERROR_OUTPUT" | grep -q "Package conflict detected: The legacy package 'giskard' is installed" || \
		(echo "Error: Expected error message not found for 'import giskard.core'" && echo "Got: $$ERROR_OUTPUT" && exit 1)
	@echo "Testing import giskard raises expected error..."
	@ERROR_OUTPUT=$$(.venv-package-conflict/bin/python -c "import giskard" 2>&1) || true; \
	echo "$$ERROR_OUTPUT" | grep -q "Package conflict detected: The legacy package 'giskard' is installed" || \
		(echo "Error: Expected error message not found for 'import giskard'" && echo "Got: $$ERROR_OUTPUT" && exit 1)
	@echo "✓ Package conflict test passed!"
	rm -rf .venv-package-conflict

lint: ## Run linting checks
	uv tool run ruff check .

format: ## Format code with ruff
	uv tool run ruff format .
	uv tool run ruff check --fix .

check-format: ## Check if code is formatted correctly
	uv tool run ruff format --check .

check-compat: ## Check Python 3.11 compatibility
	uv tool run vermin --target=3.11- --no-tips --violations .

typecheck: ## Run type checking with basedpyright
	uv tool run basedpyright --level error .

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
check: lint check-format check-compat typecheck security check-licenses ## Run all checks (lint, format, compatibility, typecheck)

all: format check test ## Format, check, and test

# CI simulation
ci: check test ## Run the same checks as CI

clean: ## Clean up build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
