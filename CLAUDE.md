# CLAUDE.md

Claude-specific notes for the **giskard-oss** monorepo. Human-oriented docs live in [README.md](README.md); agent-oriented operational notes live in [AGENTS.md](AGENTS.md).

## Project overview

- **Stack:** Python 3.12+, [uv](https://docs.astral.sh/uv/) for env/runs, [Ruff](https://docs.astral.sh/ruff/) for lint/format, [basedpyright](https://github.com/DetachHead/basedpyright) for types.
- **Layout:** Libraries under `libs/` — `giskard-core`, `giskard-agents`, `giskard-checks`, `giskard-llm`. Work from the repo root unless a task is strictly scoped to one package.

## Setup

```bash
make setup            # uv sync + install tools + pre-commit hooks
# or, for coding agents:
make setup-for-agents
```

## Commands

| Goal | Command |
|------|---------|
| Format | `make format` |
| Lint | `make lint` |
| Full gate | `make check` |
| Unit tests (all) | `make test-unit` |
| Unit tests (one package) | `make test-unit PACKAGE=giskard-checks` |
| All tests | `make test` |

Before opening or updating a PR: `make format && make check && make test-unit`.

## PR discipline

- End agent-opened PR titles with `🤖🤖🤖🤖` — required for the expedited-agent PR workflow.
- Always run `uv run` (not bare `python` or `pytest`).

---

## Core principles

**Make every change as small as possible.** Prefer deleting lines over adding them. The best fix often removes code entirely.

**Find the root cause.** No band-aids. If a symptom has an underlying cause, address that — not the symptom.

**Only touch what's necessary.** No side effects. Leave every file you didn't need to change exactly as you found it.
