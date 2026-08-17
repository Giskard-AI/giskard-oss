# Variables
LIBS := giskard-core giskard-llm giskard-agents giskard-checks giskard-scan
# The umbrella distribution built from the repo root pyproject.toml. It is not a libs/
# member and ships no code of its own -- it only pins version ranges over LIBS, so
# "test the metapackage" means "test everything it pins", i.e. all of LIBS.
METAPACKAGE := giskard
PACKAGE ?= # Optional package to test (e.g., giskard-core, giskard-agents, giskard-checks)
AGENT_NAME ?= # Optional, for setup-for-agents telemetry
REASON ?= # Optional, for setup-for-agents telemetry

# Fail loudly on a typo'd PACKAGE rather than letting it become a bogus libs/<typo>
# path that surfaces as an opaque uv chdir error several steps later.
ifneq ($(PACKAGE),)
ifeq ($(filter $(PACKAGE),$(LIBS) $(METAPACKAGE)),)
$(error Unknown PACKAGE '$(PACKAGE)'. Valid values: $(METAPACKAGE) $(LIBS))
endif
endif

# The libs every test target iterates. Unset PACKAGE and PACKAGE=$(METAPACKAGE) both
# mean the whole workspace; anything else is that single lib.
ifeq ($(filter-out $(METAPACKAGE),$(PACKAGE)),)
TEST_LIBS := $(LIBS)
else
TEST_LIBS := $(PACKAGE)
endif

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_%-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install project dependencies
	uv sync

# Ruff formatting output changes between releases, so an unpinned install can turn
# `make check` red on a tree nobody touched. Pinned for the same reason as
# LICENSECHECK_VERSION below.
RUFF_VERSION := 0.16.1

install-tools: ## Install development tools
	uv tool install ruff==$(RUFF_VERSION)
	uv tool install vermin
	uv tool install basedpyright
	uv tool install pre-commit --with pre-commit-uv

pre-commit-install: ## Setup pre-commit hooks
	uv tool run pre-commit install

setup: install install-tools pre-commit-install ## Complete development setup (install deps + tools)

setup-for-agents: setup ## Like setup + agent .env and analytics ping; optional AGENT_NAME= REASON=
	@if ! grep -qE '^(export[[:space:]]+)?GISKARD_TELEMETRY_DISABLED=' .env 2>/dev/null; then \
		echo "GISKARD_TELEMETRY_DISABLED=false" >> .env; \
	fi
	@AGENT_NAME="$(AGENT_NAME)" REASON="$(REASON)" uv run python -c "import json, os, urllib.request; from datetime import datetime, timezone; a=os.environ.get('AGENT_NAME','').strip(); r=os.environ.get('REASON','').strip(); p={**({'agent_name': a} if a else {}), **({'reason': r} if r else {})}; body=json.dumps({'api_key': 'phc_Asp36pe4X5WMqeJ4aMMV4gq5LGdGw69mdYSdEYGpbxm2', 'event': 'giskard_oss_agents_setup', 'distinct_id': a or 'giskard_oss_agent', 'properties': p, 'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'}); urllib.request.urlopen(urllib.request.Request('https://eu.i.posthog.com/i/v0/e/', data=body.encode(), headers={'Content-Type': 'application/json'}, method='POST'), timeout=30)"

# Run pytest once per lib in TEST_LIBS, passing $(1) as the pytest arguments. Each run
# must chdir into the lib (uv run --directory) rather than passing libs/<lib> as a path
# from the root: lib pytest configs carry CWD-relative settings, and
# libs/giskard-scan/pyproject.toml ignores its optional garak/lidar/deepteam integration
# trees via relative --ignore paths. Invoked from the root those paths resolve to
# nothing, the ignores silently miss, and --doctest-modules then imports modules that
# need optional extras (ModuleNotFoundError: No module named 'garak'). The trailing
# `&& ... true` chain stops at the first failing lib.
pytest-each-lib = $(foreach lib,$(TEST_LIBS),uv run --directory libs/$(lib) pytest $(1) &&) true

test: ## Run all tests (unit + functional), optional PACKAGE=<name>
	$(call pytest-each-lib)

test-unit: ## Run unit tests only (excludes functional), optional PACKAGE=<name>
	$(call pytest-each-lib,tests src -m "not functional")

test-functional: ## Run functional tests only (requires API keys), optional PACKAGE=<name> PROVIDER=<name>
	$(call pytest-each-lib,-m "functional$(if $(PROVIDER), and $(PROVIDER))")

install-no-providers: ## Install giskard-llm without provider SDKs (for no_providers tests)
	uv sync --package giskard-llm

install-minimal: ## Install with test group only (no provider SDKs, all packages)
	uv sync --only-group test

install-garak-test: ## Install garak optional extra for scan integration tests
	uv sync --group garak-test

install-deepteam-test: ## Install deepteam optional extra for scan integration tests
	uv sync --group deepteam-test

# Private package — not in pyproject/uv.lock so public CI/release never fetch it.
LIDAR_GIT ?= git+https://github.com/Giskard-AI/lidar.git@v0.2.7

install-lidar-test: ## Install lidar private dependency for scan integration tests
	uv pip install "$(LIDAR_GIT)"

test-lidar: install-lidar-test ## Run lidar integration tests
	uv run pytest libs/giskard-scan/tests/integrations/lidar -v

# Deliberately NOT pytest-each-lib: this runs under `make install-minimal` (no provider
# SDKs), so it must collect LESS than test-unit, not the same. Passing `tests src` would
# pull in src doctests that need optional extras -- giskard-checks' RegoPolicy docstring
# needs regorus, and giskard-llm's tests/test_smoke.py calls find_spec("google.genai") at
# import time. Staying at the repo root also leaves each lib's testpaths/addopts
# unapplied, which is what keeps those src trees out of collection here.
test-unit-minimal: ## Run unit tests on minimal deps (no provider SDKs), optional PACKAGE=<name>
	$(foreach lib,$(TEST_LIBS),uv run pytest libs/$(lib) -m "not functional" &&) true

test-examples: ## Run canonical examples and README snippet lint
	uv run pytest examples tools/test_lint_readme_snippets.py -q
	uv run python tools/lint_readme_snippets.py

test-no-providers: ## Run tests that verify behavior when provider SDKs are missing
	uv run pytest libs/giskard-llm -m "no_providers"

test-garak: ## Run garak integration tests (requires: make install-garak-test)
	uv run pytest libs/giskard-scan/tests/integrations/garak

test-deepteam: ## Run deepteam integration tests (requires: make install-deepteam-test)
	uv run pytest libs/giskard-scan/tests/integrations/deepteam

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
	uv run ruff format .
	uv run ruff check --fix .

check-format: ## Check if code is formatted correctly
	uv run ruff format --check .

check-compat: ## Check Python 3.12 compatibility
	uv tool run vermin --target=3.12- --no-tips --violations .

typecheck: ## Run type checking with basedpyright
	uv tool run basedpyright --level error .

security: ## Check for security vulnerabilities
	uv run pip-audit --skip-editable

# Run licensecheck INSIDE the synced project env (uv run --with, not uvx): it reads
# each package's version from the installed env via importlib, so output is pinned to
# uv.lock instead of whatever PyPI resolves to at runtime. Pinned for reproducibility.
LICENSECHECK_VERSION := 2026.0.8
LICENSECHECK := uv run --with licensecheck==$(LICENSECHECK_VERSION) licensecheck --license MIT
# Scan scope for the default license/notices gate. Keep this aligned with what
# `pip install giskard[full]` actually pulls: optional scan extras `garak` and
# `deepteam` are intentionally omitted here. Their transitive trees are large
# (and a frequent CVE/audit surface), and we do not ship those deps by default.
# Upstream projects (for when this choice is revisited):
#   garak    — https://github.com/NVIDIA/garak
#   deepteam — https://github.com/confident-ai/deepteam
# To include them later: `LICENSECHECK_EXTRAS := full garak deepteam` (space-
# separated; --extras is nargs, so a comma-separated value is one literal name
# and silently scans nothing), after installing those extras into the env.
LICENSECHECK_EXTRAS := full
# Permissive licenses that licensecheck cannot parse, so it falls back to UNKNOWN and
# fails them: datetime/zope-interface are ZPL-2.1 (BSD-style), mistralai and
# sentencepiece are Apache-2.0 but publish no license metadata. Historically
# reached via the garak tree when that extra was in scope; kept so a future
# re-enable of garak/deepteam does not trip the gate on metadata gaps alone.
# These are detection gaps, not license conflicts -- do not read this as suppressing a
# genuine copyleft warning.
LICENSECHECK_IGNORE := datetime zope-interface mistralai sentencepiece
# Skip the workspace libs themselves (LIBS) so the notices list only third-party packages.
LICENSECHECK_FLAGS := --extras $(LICENSECHECK_EXTRAS) --skip-dependencies $(LIBS) \
	--ignore-packages $(LICENSECHECK_IGNORE)
# licensecheck markdown output is not byte-stable (trailing whitespace, blank-line
# runs), so canonicalize it before writing/diffing: strip trailing whitespace and
# collapse consecutive blank lines.
NORMALIZE := sed -e 's/[[:space:]]*$$//' | awk '/^$$/{blank++; next} {for(i=0;i<blank;i++) print ""; blank=0; print}'

generate-notices: ## Generate THIRD_PARTY_NOTICES.md
	$(LICENSECHECK) $(LICENSECHECK_FLAGS) \
		--format markdown --hide-output-parameters SIZE --file /dev/stdout \
		| $(NORMALIZE) > THIRD_PARTY_NOTICES.md

check-licenses: ## Check for licenses
	$(LICENSECHECK) $(LICENSECHECK_FLAGS) --show-only-failing --zero

check-notices: ## Check that THIRD_PARTY_NOTICES.md is up to date (run make generate-notices if this fails)
	@TMPFILE=$$(mktemp); trap 'rm -f "$$TMPFILE"' EXIT; \
	$(LICENSECHECK) $(LICENSECHECK_FLAGS) \
		--format markdown --hide-output-parameters SIZE --file /dev/stdout \
		| $(NORMALIZE) > $$TMPFILE && \
	if ! diff $$TMPFILE THIRD_PARTY_NOTICES.md; then \
		echo "THIRD_PARTY_NOTICES.md is out of date. Run: make generate-notices"; \
		exit 1; \
	fi

check-extra-pins: ## Assert root pyproject lower bounds match workspace member versions
	uv run python tools/check_extra_pins.py

check: lint check-format check-compat typecheck security check-licenses check-notices check-extra-pins ## Run all checks

clean: ## Clean up build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
