---
name: giskard-oss-pr-ready
description: Describes the merge-ready workflow for the giskard-oss monorepo using the repository root Makefile and the same steps as GitHub Actions—format, check, unit tests, optional pre-commit. Use when preparing or updating a pull request, confirming CI will pass, running pre-push validation, or when the user asks for a branch to be merge-ready.
---

# giskard-oss — PR ready

## Context

Work from the **repository root** (`giskard-oss/`). The monorepo `Makefile` lives there.

Main CI (`.github/workflows/ci.yml`) runs: `make install install-tools`, then `make check`, then `make test-unit PACKAGE=<each of giskard-core, giskard-agents, giskard-checks>` on Python 3.12–3.14.

## Merge-ready checklist

Run in order:

1. **`make format`** — Ruff format + fix where applicable.

2. **`make check`** — Lint, format check, Python 3.12 compat (vermin), basedpyright, pip-audit, license check. This matches the `lint` job in CI (after `make install install-tools`).

3. **`make test-unit`** — All packages, or narrow scope:
   - `make test-unit PACKAGE=giskard-checks` (same pattern for `giskard-core`, `giskard-agents`)

4. **Optional — pre-commit** — If hooks are installed: `pre-commit run`. First-time setup: `make pre-commit-install` (part of `make setup`).

## Dependencies

If checks fail from missing tools, run **`make install`** and **`make install-tools`** (CI runs both before `make check`).

## Functional tests

`make test-functional` (and full `make test`) hit live APIs and are not the default PR gate in `ci.yml` (they run in `.github/workflows/integration-tests.yml` with repo secrets). Run them only when the change requires `@pytest.mark.functional` coverage.

### `.env` setup (local)

1. At the **repository root**, create a `.env` file (already listed in `.gitignore`—do not commit secrets).
2. Define credentials for the models your tests use. The integration workflow sets:
   - `GEMINI_API_KEY` — required for the default Gemini routes used by functional tests.
   - `TEST_MODEL` — optional; default `gemini/gemini-2.0-flash` (see `libs/giskard-agents/tests/conftest.py`).
   - `TEST_EMBEDDING_MODEL` — optional; default `gemini/gemini-embedding-001`.
3. The repo does not auto-load `.env` in pytest—**export variables into the shell** before running tests, for example:

   ```bash
   set -a && source .env && set +a
   make test-functional PACKAGE=giskard-agents
   ```

   Use valid shell assignments in `.env` (e.g. `GEMINI_API_KEY=...`). If you point `TEST_MODEL` / `TEST_EMBEDDING_MODEL` at another provider, add whatever API keys that provider expects (LiteLLM / the underlying SDK).

## Library-specific rules

For package conventions (API patterns, tests layout), read `libs/<package>/.cursor/rules/`. Makefile targets are always invoked from the repository root as listed above.
