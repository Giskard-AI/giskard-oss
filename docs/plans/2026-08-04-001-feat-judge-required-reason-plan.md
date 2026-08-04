---
title: Judge Required Reason - Plan
type: feat
date: 2026-08-04
topic: judge-required-reason
artifact_contract: ce-unified-plan/v1
artifact_readiness: requirements-only
product_contract_source: ce-brainstorm
execution: code
---

# Judge Required Reason - Plan

## Goal Capsule

- **Objective:** Make every `LLMCheckResult` carry a required, non-blank `reason` on pass and fail, and align built-in judge prompts so the model is always asked for a clear reason.
- **Product authority:** giskard-checks judge / `BaseLLMCheck` structured output contract; adjacent display and custom-prompt judges are out of scope.
- **Open blockers:** none.

## Product Contract

### Summary

Judge structured output must always include a non-blank explanation for the verdict.
Built-in judge prompts must ask for that clear reason on every evaluation (pass and fail).

### Problem Frame

Today `reason` is optional on `LLMCheckResult`.
A pass can land with no explanation (or a generic fallback), so consumers cannot tell a justified pass from a vacuous one.
Some built-in prompts constrain how to reason about failures but never require a clear reason on every decision.

### Key Decisions

- **Require non-blank `reason` on the shared result model for both pass and fail** (session-settled: user-directed — chosen over pass-only flexibility: same model, same trust bar). Governs R1, R2.
- **Enforce via schema validation, not post-hoc rewriting of `CheckResult`** (session-settled: user-directed — chosen over optional reason + display/UI changes: reject incomplete judge output at parse time). Governs R1, R2.
- **Non-blank only** (session-settled: user-directed — chosen over min-length or ban on vague phrases like "ok": smallest change that makes vacuous passes untrustworthy). Governs R2.
- **Update built-in judge prompts to always request a clear reason** (session-settled: user-directed — added at confirmation). Governs R4.
- **Do not change rich-console pass display in this work** (session-settled: user-directed — chosen over "surface pass reasons" as primary pain). Governs Scope Boundaries.

### Requirements

**Schema**

- R1. `LLMCheckResult.reason` is required (not optional / not nullable).
- R2. Empty or whitespace-only `reason` values are invalid under the same schema validation as R1.

**Runtime behavior**

- R3. When structured output fails R1/R2, the check must not become a silent PASS; invalid output follows the existing structured-output validation failure path (retry, then error/failure upstream).

**Prompts**

- R4. Every built-in judge prompt template instructs the model to provide a clear `reason` for the evaluation decision on both pass and fail, without weakening existing pass/fail criteria.

**Docs / API surface**

- R5. Public docs and descriptions that still call judge `reason` optional are updated to match the required contract.

### Acceptance Examples

- AE1. Covers R1, R2. **Given** judge structured output with `passed: true` and missing/`null`/`""`/`"   "` reason, **When** the output is validated, **Then** validation fails and the run does not record a trustworthy PASS with empty reason.
- AE2. Covers R1. **Given** judge structured output with `passed: false` and a non-blank reason, **When** the output is validated, **Then** validation succeeds and the failure carries that reason.
- AE3. Covers R4. **Given** each built-in judge prompt template, **When** inspected, **Then** it asks for a clear reason on the evaluation decision for both outcomes (pass and fail).
- AE4. Covers R3. **Given** repeated invalid blank reasons from the model under structured output, **When** retries are exhausted, **Then** the check outcome is not PASS.

### Scope Boundaries

**In scope**

- `LLMCheckResult` required non-blank `reason`
- Built-in judge prompt templates under `libs/giskard-checks/src/giskard/checks/prompts/judges/`
- Tests and public wording that assume optional/`None` reason
- Fallbacks that paper over missing reason once the field is always present

**Out of scope**

- Showing pass reasons in the rich console / report UI
- Rejecting low-quality or boilerplate wording beyond blank/whitespace
- Custom `Judge` prompts supplied by callers
- Custom `output_type` models that do not use `LLMCheckResult`

### Dependencies / Assumptions

- Assumption: structured-output `ValidationError` already retries then surfaces upstream as a non-PASS outcome for LLM checks; this work relies on that path rather than inventing a new status.
- Assumption: user-authored `Judge` prompts remain the caller's responsibility; schema validation still rejects blank reasons even if their prompt forgets to ask.

### Sources / Research

- Current optional field and pass/fail mapping: `libs/giskard-checks/src/giskard/checks/judges/base.py`
- Built-in prompts: `libs/giskard-checks/src/giskard/checks/prompts/judges/` (`conformity`, `toxicity`, `answer_relevance`, `groundedness`, `contradiction`)
- Structured-output validation retries: `libs/giskard-agents/src/giskard/agents/workflow.py`
- Console currently blanks PASS messages: `libs/giskard-checks/src/giskard/checks/core/result.py` (deliberately unchanged here)
