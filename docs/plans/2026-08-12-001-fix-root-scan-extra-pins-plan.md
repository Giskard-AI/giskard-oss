---
title: "Align root giskard-scan extras and pin check - Plan"
type: fix
date: 2026-08-12
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
origin: https://github.com/Giskard-AI/giskard-oss/issues/2731
---

# Align root giskard-scan extras and pin check - Plan

## Goal Capsule

**Objective.** Make root `giskard[scan|garak|deepteam]` lower bounds match `libs/giskard-scan` `1.0.0b4`, and add a CI-gated check so root↔member pin drift cannot recur silently.

**Authority.** Issue #2731 is the product authority; this plan owns the mechanism.

**Out of scope.** Inter-lib pins under `libs/*/pyproject.toml` (already verified correct). Changing published package versions. Aggregator extras that only expand other extras (`full`, `all-llms`, `all-checks`).

**Stop when.** The three root pins equal `1.0.0b4`, `make check-extra-pins` fails on intentional drift and passes on the fixed tree, and `make check` includes that target.

## Product Contract

### Summary

Root `pyproject.toml` still pins `giskard-scan` extras at `>=1.0.0b2` while the workspace member is `1.0.0b4`. Users installing `giskard[scan]` can resolve an older scan than CI tests. Fix the pins and add a durable equality check.

### Problem Frame

Version bumps of workspace members have been applied without matching root-extra lower bounds. Review does not catch this; CI never exercises the stale combination.

### Requirements

- R1. Root extras `scan`, `garak`, and `deepteam` use lower bound `1.0.0b4` for `giskard-scan` (with or without extras), keeping `<2`.
- R2. A check asserts that every root `dependencies` / `optional-dependencies` requirement naming a workspace member has a `>=` lower bound equal to that member's `version`.
- R3. The check runs as part of the existing `make check` / CI lint path so drift fails PRs.
- R4. Aggregator-only requirements that do not name a workspace member (e.g. `giskard[all-llms,...]`) are ignored by the check.

### Scope Boundaries

- In: root `pyproject.toml` pins; Makefile wiring; a small repo-root check script.
- Out: bumping `giskard-scan` itself; changing inter-lib pins; rewriting release tooling.

### Sources

- Issue #2731
- `pyproject.toml` optional-dependencies lines 40–44
- `libs/giskard-scan/pyproject.toml` `version = "1.0.0b4"`
- `.github/workflows/ci.yml` lint job runs `make check`

## Planning Contract

### Key Technical Decisions

- KTD1. Implement the durable check as `tools/check_extra_pins.py` invoked by a new `check-extra-pins` Makefile target, and add that target to the `check` dependency list. (Chosen over an inline Makefile recipe: parsing TOML + PEP 508 needs real Python; chosen over a pytest: this is a repo gate like `check-notices`, not a package unit test.)
- KTD2. Parse requirements with `packaging.requirements.Requirement` and TOML with stdlib `tomllib`. For each requirement whose name is a workspace member (keys under `[tool.uv.sources]` with `workspace = true`, or equivalently `libs/*/pyproject.toml` project names), require exactly one `>=` lower bound and assert it equals the member `version` string.
- KTD3. Scope the check to root `project.dependencies` and `project.optional-dependencies` only (issue-verified drift set). Do not scan `libs/*/pyproject.toml` in this change.
- KTD4. Fix the three stale pins in the same change as the check so `make check-extra-pins` is green on the resulting tree.

### Assumptions

- `packaging` remains importable via the synced uv env (`uv run`).
- Upper bound `<2` is intentional and not asserted by the new check.

### Sequencing

1. U1 pin fix
2. U2 check script + Makefile/CI wiring
3. Verify green, then red-on-drift smoke

## Implementation Units

### U1. Align root giskard-scan extra pins

**Goal.** Satisfy R1.

**Requirements.** R1

**Files.** `pyproject.toml`

**Approach.** Replace `>=1.0.0b2` with `>=1.0.0b4` on `scan`, `garak`, and `deepteam` only. Leave comments and other extras unchanged.

**Test scenarios.**

- T1. After edit, those three strings contain `giskard-scan`…`>=1.0.0b4,<2` (with extras where applicable).
- T2. No other root optional-dependency lines change.

**Verification.** Visual/diff review; later U2 check passes.

**Dependencies.** None.

### U2. Root extra pin equality gate

**Goal.** Satisfy R2–R4.

**Requirements.** R2, R3, R4

**Files.** `tools/check_extra_pins.py`, `Makefile`

**Approach.** Script loads root `pyproject.toml`, builds `{name: version}` from workspace members under `libs/*/pyproject.toml`, walks root dependency lists, skips non-member names, compares `>=` lower bound to member version, exits non-zero with a clear message listing mismatches. Makefile: `check-extra-pins` runs `uv run python tools/check_extra_pins.py`; `check` depends on it (alongside existing lint/format/compat/typecheck/security/license targets).

**Test scenarios.**

- T1. On the fixed tree, `make check-extra-pins` exits 0.
- T2. Temporarily restore one pin to `b2`, rerun, expect non-zero and a message naming `giskard-scan` / expected `1.0.0b4` / found `1.0.0b2`; restore the fix.
- T3. Aggregator extra `full` does not cause a false failure.
- T4. `make check` still invokes the new target (listed in the `check` recipe).

**Verification.** Commands in Verification Contract.

**Dependencies.** U1 (so the check passes on the branch tip).

## Verification Contract

- `make check-extra-pins` — must pass.
- Intentional drift smoke (local, revert after): edit one pin back to `b2`, confirm failure, restore.
- `make format && make check` — full quality gate (network-dependent license/security steps may fail offline; pin check itself must pass regardless).
- No package unit-test matrix change required; behavior is repo-tooling, not library runtime.

## Definition of Done

- R1–R4 satisfied.
- U1 and U2 complete; drift smoke proved.
- Abandoned experimental check variants removed from the tree.
- Conventional commit(s); PR references #2731.

## Review

Implemented on `cursor/fix-root-scan-extra-pins-6e35` as PR #2737.

- U1: root `scan` / `garak` / `deepteam` → `>=1.0.0b4,<2`.
- U2: `tools/check_extra_pins.py` + `make check-extra-pins` in `check`.
- Verification: intentional `b2` drift fails; restored tree + full `make check` pass.
