# Variables
LIBS := giskard-core giskard-agents giskard-checks
PACKAGE ?= # Optional package to test (e.g., giskard-core, giskard-agents, giskard-checks)

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_%-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install project dependencies
	uv sync

install-tools: ## Install development tools
	uv tool install ruff
	uv tool install vermin
	uv tool install basedpyright
	uv tool install pre-commit --with pre-commit-uv

pre-commit-install: ## Setup pre-commit hooks
	uv tool run pre-commit install

setup: install install-tools pre-commit-install ## Complete development setup (install deps + tools)

test: ## Run all tests (unit + functional), optional PACKAGE=<name>
ifdef PACKAGE
	uv run pytest libs/$(PACKAGE)
else
	uv run pytest libs/
endif

test-unit: ## Run unit tests only (excludes functional), optional PACKAGE=<name>
ifdef PACKAGE
	uv run pytest libs/$(PACKAGE) -m "not functional"
else
	$(foreach lib,$(LIBS),uv run pytest libs/$(lib) -m "not functional" &&) true
endif

test-functional: ## Run functional tests only (requires API keys), optional PACKAGE=<name>
ifdef PACKAGE
	uv run pytest libs/$(PACKAGE) -m "functional"
else
	$(foreach lib,$(LIBS),uv run pytest libs/$(lib) -m "functional" &&) true
endif

test-package-conflict: ## Test package conflict with giskard legacy package installed
	@echo "Testing package conflict..."
	@echo "Creating virtual environment..."
	uv venv --seed -p 3.12 .venv-package-conflict
	@echo "Installing giskard..."
	.venv-package-conflict/bin/pip install giskard
	@echo "Installing giskard-core..."
	.venv-package-conflict/bin/pip install libs/giskard-core
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
	uv run ruff check .

format: ## Format code with ruff
	uv tool run ruff format .
	uv tool run ruff check --fix .

check-format: ## Check if code is formatted correctly
	uv tool run ruff format --check .

check-compat: ## Check Python 3.12 compatibility
	uv tool run vermin --target=3.12- --no-tips --violations .

typecheck: ## Run type checking with basedpyright
	uv tool run basedpyright --level error .

security: ## Check for security vulnerabilities
	uv run pip-audit --skip-editable

generate-licenses: ## Generate licenses
	uv tool run licensecheck --license MIT \
		--format markdown --file THIRD_PARTY_NOTICES.md \
		--skip-dependencies giskard-agents giskard-checks giskard-core

check-licenses: ## Check for licenses
	uv tool run licensecheck --license MIT \
		--show-only-failing --zero \
		--skip-dependencies giskard-agents giskard-checks giskard-core

check: lint check-format check-compat typecheck security check-licenses ## Run all checks

clean: ## Clean up build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
