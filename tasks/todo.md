# Collapse duplicate agents.Error into core.Error (#2742)

## Approach

1. Add failing tests proving `giskard.agents.Error is not giskard.core.Error`.
2. Replace the duplicate BaseModel in `agents/errors/serializable.py` with a re-export of `giskard.core.Error`.
3. Keep public exports (`from giskard.agents import Error`) unchanged.
4. Verify with `make format && make check && make test-unit PACKAGE=giskard-agents`.

## Status

- [x] Re-export implemented
- [x] Identity regression tests added
- [x] Verification passed
- [x] PR opened (closes #2742)

## Review / results

- Diff is two files only; no giskard-core source changes.
- Pre-fix: identity tests failed (`AgentsError is CoreError`, `isinstance` across imports).
- Post-fix: 131 passed / 2 skipped (agents); 41 passed (core).
- Breaking only for callers that required the two classes to be distinct objects.
