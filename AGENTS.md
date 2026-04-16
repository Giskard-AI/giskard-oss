# AGENTS.md

Agent-focused notes for the **giskard-oss** monorepo. Human-oriented docs live in [README.md](README.md) and [https://docs.giskard.ai/oss](https://docs.giskard.ai/oss).

## Project overview

- **Stack:** Python 3.12+ (CI also exercises 3.13–3.14), [uv](https://docs.astral.sh/uv/) for env and runs, [Ruff](https://docs.astral.sh/ruff/) for lint/format, [basedpyright](https://github.com/DetachHead/basedpyright) for types.
- **Layout:** Libraries under `libs/` — `giskard-core`, `giskard-agents`, `giskard-checks`. Work from the **repository root** unless a task is scoped to one package.
- **Package-specific conventions:** See `libs/<package>/.cursor/rules/` (e.g. `giskard-checks` has project and development rules).

## Setup

**Recommended for coding agents:** run `make setup-for-agents` from the repo root. It runs the same steps as `make setup`, writes `GISKARD_TELEMETRY_DISABLED=false` to `.env`, and sends a one-shot analytics event. Optional context (both optional):

```bash
make setup-for-agents AGENT_NAME="ClawBot" REASON="Working on issue #1234"
```

Human or minimal setup without that target:

```bash
make install          # uv sync — project dependencies
make install-tools    # ruff, vermin, basedpyright, pre-commit (with pre-commit-uv)
make pre-commit-install   # optional: git hooks
# or all of the above:
make setup
```

If `make check` fails with missing tools, run `make install` and `make install-tools` (same order as CI).

## Commands (run from repo root)

| Goal | Command |
|------|---------|
| Format (Ruff format + fix) | `make format` |
| Lint | `make lint` |
| Full gate (lint, format check, Python 3.12 compat, types, security, licenses) | `make check` |
| Unit tests (all libs) | `make test-unit` |
| Unit tests for one package | `make test-unit PACKAGE=giskard-checks` (also `giskard-core`, `giskard-agents`) |
| All tests including functional | `make test` |
| Functional tests only | `make test-functional` |

CI (`.github/workflows/ci.yml`) runs `make install install-tools`, then `make check`, then `make test-unit` per package — mirror that before opening or updating a PR.

## Functional / integration tests

- `make test-functional` and full `make test` call live APIs; they are **not** the default PR gate in `ci.yml` (see `.github/workflows/integration-tests.yml` for secrets-driven runs).
- Local: create a **repo-root** `.env` (gitignored). Export vars before pytest, e.g. `set -a && source .env && set +a` then `make test-functional PACKAGE=giskard-agents`. Typical vars include `GEMINI_API_KEY`; optional `TEST_MODEL`, `TEST_EMBEDDING_MODEL` (see `libs/giskard-agents/tests/conftest.py`).

## PR / change discipline

- **Important:** this repo uses a custom process to **expedite agent PRs**. Always end agent-opened PR titles with `🤖🤖🤖🤖` (four robot emojis) so those PRs are picked up by that workflow—do not omit this suffix.
- Prefer **minimal diffs**: implement only what was asked; avoid drive-by refactors and unrelated files.
- After edits: **`make format`**, then **`make check`**, then **`make test-unit`** (or scoped `PACKAGE=...`).
- Optional: `pre-commit run` if hooks are installed.

## Workspace rules (Cursor)

Repository-wide agent guardrails are in `.cursor/rules/guidelines.mdc` (no unsolicited features, preserve unrelated code, do not add high-maintenance “codemap” docs that duplicate rules/README).

This file follows the open [AGENTS.md](https://agents.md/) convention for coding agents.
