---
title: Structural Unevaluable Outcomes as ERROR - Plan
type: fix
date: 2026-08-03
topic: not-structural-error
artifact_contract: ce-unified-plan/v1
artifact_readiness: requirements-only
product_contract_source: ce-brainstorm
execution: code
---

# Structural Unevaluable Outcomes as ERROR - Plan

## Goal Capsule

- **Objective:** Stop `Not` from turning structural “could not evaluate” outcomes into a green suite by classifying those outcomes as `ERROR` on `main` (where no `evaluable` field exists), and ship that fix as a fresh PR that supersedes #2658.
- **Product authority:** This plan owns the check-result status contract for structural unevaluable cases and the delivery vehicle (fresh PR from `main`). Broader composition redesign is not active scope.
- **Open blockers:** None for planning. Exact builtin branch inventory is deferred to planning.

---

## Product Contract

### Summary

Remap structural “could not evaluate” check outcomes from `FAIL` to `ERROR` so `Not` cannot invert them into `PASS`. Do not add a public `evaluable` (or similar) field. Implement on current `main` and open a fresh PR that supersedes #2658 while closing #2637.

### Problem Frame

`Not` inverts every `FAIL` into `PASS`. Built-ins today use `FAIL` for both “assertion ran and did not hold” and “assertion could not run” (missing key, wrong type for configured mode, unsupported comparison). A typo’d or renamed key under `Not(...)` therefore reports green. Commit history shows missing-key → `FAIL` as a v3 convention, not a documented requirement; `ERROR` was reserved for exceptions / unexpected conditions and already passes through `Not` unchanged. PR #2658 proposed an additive `evaluable` flag on a branch; that field does not exist on `main`, and this work rejects that API in favor of the existing status enum.

### Key Decisions

- KD1. **Reuse `ERROR` instead of a new public field** (session-settled: user-directed — chosen over `evaluable` on #2658: avoid permanent API carrying cost). Governs R1, R6.
- KD2. **Structural unevaluable outcomes are `ERROR`** (session-settled: user-approved — chosen over keeping them as `FAIL`: FAIL means assertion evaluated and failed; ERROR means it could not run). Governs R2, R3, R4.
- KD3. **`AnyOf` fail-loud on structural `ERROR` is correct** (session-settled: user-directed — chosen over continuing after ERROR like FAIL: bad config should not silently fall through). Governs R5.
- KD4. **Ship as a fresh PR from `main`; supersede #2658** (session-settled: user-approved — chosen over rewriting #2658: different contract, no review thread to preserve, baseline has no `evaluable`). Governs R7.

### Requirements

**Status contract**

- R1. No new public result field is introduced to mark unevaluable outcomes; the existing `CheckStatus` values remain the only status axis.
- R2. When a check cannot evaluate its configured assertion because a required value is missing (including `NoMatch` / “no value found for key”), the result status is `ERROR`, not `FAIL`.
- R3. When a check cannot evaluate because the value’s type is incompatible with the configured mode (for example collection `match` against a non-collection) or the comparison is unsupported for the value types, the result status is `ERROR`, not `FAIL`.
- R4. Soft structural cases that prevent scoring the intended assertion (required text empty, value not a string when a string is required, and similar “cannot run the check as configured”) use `ERROR` under the same rule as R2–R3. Pure assertion-false outcomes remain `FAIL`.

**Composition behavior**

- R5. Existing composition semantics for `ERROR` are preserved: `Not` passes `ERROR` through without inversion; `AnyOf` short-circuits and returns the `ERROR`; `AllOf` stops on non-pass. No special-case bypass for structural errors.

**Compatibility and delivery**

- R6. Behavior change for consumers that assert `FAIL` / `.failed` on structural cases is accepted and called out as a breaking behavior change in release notes / changelog.
- R7. Implementation lands on a branch from current `main` as a new PR that closes #2637 and supersedes #2658 (close or comment with pointer). Do not merge the `evaluable` approach.

### Key Flows

- F1. Missing key under `Not`
  - **Trigger:** `Not(Equals(key=missing, ...))` (or equivalent) runs against a trace without that key.
  - **Steps:** Inner check returns `ERROR` with a missing-key message; `Not` returns that result unchanged.
  - **Outcome:** Suite/scenario does not pass; status is `ERROR`, not `PASS`.
  - **Covered by:** R2, R5

- F2. Structural miss inside `AnyOf`
  - **Trigger:** `AnyOf` first arm hits a structural unevaluable condition; later arms might otherwise pass.
  - **Steps:** First arm returns `ERROR`; `AnyOf` returns that `ERROR` immediately.
  - **Outcome:** Compound check errors fail-loud; later arms are not run.
  - **Covered by:** R5

- F3. True assertion failure still inverts
  - **Trigger:** Inner check evaluates successfully and fails the assertion (value present, comparable, wrong).
  - **Steps:** Inner returns `FAIL`; `Not` inverts to `PASS`.
  - **Outcome:** Negation semantics unchanged for evaluable failures.
  - **Covered by:** R4

### Acceptance Examples

- AE1. Missing metadata key under `Not`
  - **Covers:** R2, R5
  - **Given:** Trace with outputs but no `trace.last.metadata.nope`
  - **When:** `Not(Equals(key="trace.last.metadata.nope", expected_value="x"))` runs
  - **Then:** Result status is `ERROR` (not `PASS` or `FAIL`); message still names the missing key

- AE2. `match` type mismatch under `Not`
  - **Covers:** R3, R5
  - **Given:** `trace.last.outputs` is a string
  - **When:** `Not(Equals(key="trace.last.outputs", expected_value="x", match="any"))` runs
  - **Then:** Result status is `ERROR`; not inverted to `PASS`

- AE3. Evaluable failure still inverts
  - **Covers:** R4
  - **Given:** Outputs equal `"the answer"`
  - **When:** `Not(Equals(key="trace.last.outputs", expected_value="other"))` runs
  - **Then:** Result status is `PASS` (inner `FAIL` inverted)

### Scope Boundaries

- Deferred for later: optional redesign of `AnyOf` ERROR short-circuit; new status values beyond the existing enum; Hub-specific presentation tweaks beyond existing ERROR handling.
- Outside this work: docs-only mitigation; keeping structural cases as `FAIL` with any parallel flag; merging or continuing the `evaluable` implementation on #2658.

### Dependencies / Assumptions

- Assumption: `CheckResult.error` / `CheckStatus.ERROR` already mean “exception or unexpected condition,” which covers structural unevaluable cases without a new status enum value.
- Assumption: External consumers treating missing-key as `FAIL` are few enough that a changelog’d break is acceptable (library still young / v3 rewrite context).
- Dependency: Issue #2637 remains the user-facing bug; #2658 is superseded, not the implementation base.

### Outstanding Questions

**Deferred to Planning**

- Which exact builtin branches today return `FAIL` for structural reasons and must flip to `ERROR` (inventory across comparison, text matching, JSON, Rego, NLP, semantic similarity, etc.)?
- How to phrase the changelog / breaking-behavior note for JUnit (`failure` → `error`) and suite counters?
- Contributor skill / convention text that currently says `NoMatch` → `CheckResult.failure` must be updated in the same change set — confirm all doc touchpoints during planning.

### Sources / Research

- Issue: https://github.com/Giskard-AI/giskard-oss/issues/2637
- Superseded PR (evaluable approach, not the base): https://github.com/Giskard-AI/giskard-oss/pull/2658
- Origin of missing-key → `FAIL`: commit `5097937b4` (#2229 v3 rewrite); no documented FAIL-vs-ERROR debate in that PR
- Composition ERROR passthrough / short-circuit: `libs/giskard-checks/src/giskard/checks/builtin/composition.py` (`Not`, `AnyOf`)
- Status aggregation / JUnit: `libs/giskard-checks/src/giskard/checks/core/result.py`, `libs/giskard-checks/src/giskard/checks/export/junit.py`
- Baseline: current `main` has no `evaluable` field on `CheckResult`
