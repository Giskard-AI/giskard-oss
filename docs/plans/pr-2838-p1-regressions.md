# PR 2838 P1 Regression Fixes

## Goal

Prevent missing SOM evidence from becoming an invertible model verdict, and make
`judge` the sole source of truth while preserving the legacy `generator` API.

## Plan

- [x] Add failing regression tests for SOM evidence errors, direct-value judge
  checks, `Not`, judge/generator assignment, clearing, and `model_copy` updates.
- [x] Represent missing SOM evidence as an evaluation error and convert that
  error to `CheckStatus.ERROR` at the LLM-check boundary.
- [x] Build a minimal current-turn trace when explicit check values are used
  without a conversation trace.
- [x] Replace the stored legacy generator mirror with a compatibility property
  backed by `judge`, including atomic assignment and copy semantics.
- [x] Run formatting, checks, and affected package unit tests.
- [x] Complete spec-compliance and code-quality review passes.

## Review

Implemented both P1 fixes with targeted regression coverage. The spec review
found and closed explicit-`None` alias edge cases. The code-quality review found
no remaining defects; the simplification pass removed duplicate default-judge
resolution and an eager trace allocation.

Verification:

- `make format`: passed.
- `make test-unit PACKAGE=giskard-core`: 89 passed.
- `make test-unit PACKAGE=giskard-agents`: 191 passed, 2 skipped, 16 deselected.
- `make test-unit PACKAGE=giskard-checks`: 1046 passed, 4 skipped.
- Scoped `basedpyright --level error libs/giskard-checks`: 0 errors.
- `make check`: lint and format passed; stopped on the four pre-existing
  `giskard-scan` integration type errors recorded in the QA report.
- Live TypeSafe smoke test: not run because `TYPESAFE_API_KEY` is unavailable.
