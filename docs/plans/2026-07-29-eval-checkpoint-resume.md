# Eval Checkpoint / Resume Implementation Plan

> **For agentic workers:** Use executing-plans or implement task-by-task. Steps use checkbox syntax.

**Goal:** Opt-in, parallel-safe JSONL checkpointing for `Suite.run` and `generate_suite`, with resume and partial artifacts ([#2657](https://github.com/Giskard-AI/giskard-oss/issues/2657)).

**Architecture:** `RunStore` in `giskard-checks` appends events to `events.jsonl` under an asyncio lock; `Suite.run` and `generate_suite` opt in via kwargs/env; stable `checkpoint_id` in scenario annotations.

**Tech Stack:** Python 3.12+, Pydantic v2, asyncio, stdlib `json`/`pathlib` only (no new deps).

**Spec:** `docs/specs/2026-07-29-eval-checkpoint-resume-design.md`

**Branch / worktree:** `feat/eval-checkpoint-resume` @ `.claude/worktrees/eval-checkpoint-resume`

## Global Constraints

- Opt-in only; default behavior unchanged when no checkpoint dir.
- No breaking public API changes (new optional kwargs only).
- Scenario-level events in v1; schema_version for future event types.
- Parallel-safe: locked append + flush; resume by ID not order.
- NumPy-style public docstrings in this package.

## File map

| File | Role |
|---|---|
| `libs/giskard-checks/src/giskard/checks/utils/checkpoint.py` | `RunStore`, env resolution, fingerprint helpers, errors |
| `libs/giskard-checks/tests/utils/test_checkpoint.py` | Unit tests for store |
| `libs/giskard-checks/src/giskard/checks/scenarios/suite.py` | Wire checkpoint into `Suite.run` |
| `libs/giskard-checks/tests/core/test_suite_checkpoint.py` | Suite run + resume + parallel |
| `libs/giskard-scan/src/giskard/scan/catalog.py` | Wire into `generate_suite` |
| `libs/giskard-scan/tests/generators/test_catalog_checkpoint.py` | Generate resume |
| `libs/giskard-scan/src/giskard/scan/vulnerability.py` / `quality.py` | Passthrough via `generate/` + `run/` subdirs |
| `docs/specs/...` + `docs/plans/...` | Land design with PR |

---

### Task 1: RunStore — done

- [x] Tests + `RunStore` + `resolve_checkpoint_options`
- [x] Exported from `giskard.checks` (scan import-surface rule)

### Task 2: Suite.run — done

- [x] Serial + parallel append/resume; deterministic `checkpoint_id`

### Task 3: generate_suite — done

- [x] Generator-level resume; `scenario_generated` + `generator_completed` events

### Task 4: Scan passthrough — done

- [x] `vulnerability_scan` / `quality_scan` (no `third_party_scan` on this base)

### Task 5: Verify / ship

- [x] Targeted unit tests green (store, suite, generate, scan spies)
- [ ] Full `make check` + `make test-unit PACKAGE=giskard-checks|giskard-scan` before PR
- [ ] Commit + open PR linked to #2657
