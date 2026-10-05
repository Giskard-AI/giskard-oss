---
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-plan-bootstrap
execution: code
date: 2026-10-05
---

# fix: make TypeSafe SOM transport local-only

## Goal Capsule

**Objective:** Loading and running untrusted scenario data cannot disclose an environment secret through a TypeSafe SOM request.

**Means:** Preserve portable judge/model identity while removing transport and credential-selection fields from serialized TypeSafe configuration.

**Authority:** The user request and `docs/specs/typesafe-som-trusted-transport.md` define behavior; this plan defines implementation and verification.

**Stop conditions:** Stop and re-plan if safe judge round trips require accepting transport fields, or if a supported gateway cannot be configured through local environment variables.

## Product Contract

### Summary

Make the TypeSafe request destination and credential source trusted local configuration rather than scenario-controlled model data.

### Problem Frame

`Scenario.model_validate_json` reconstructs nested TypeSafe SOM models. Their current `base_url` and `api_key_env` fields allow scenario authors to choose where a bearer token is sent and which local environment value supplies it.

### Requirements

- **R1:** Scenario/check JSON cannot control the TypeSafe request URL.
- **R2:** Scenario/check JSON cannot select an environment variable for credentials.
- **R3:** Local environment configuration retains direct API and gateway support.
- **R4:** TypeSafe kind/model selection continues to round-trip.
- **R5:** Payloads containing transport overrides fail validation.

### Scope Boundaries

In scope: TypeSafe SOM transport configuration and scenario/check regression coverage. Out of scope: removing all judge serialization, changing other providers, or adding a trusted-deserialization context protocol.

## Planning Contract

### Key Technical Decisions

- **KTD1:** Delete `base_url` and `api_key_env` from the TypeSafe Pydantic model. This closes every deserialization entry point rather than relying on callers to propagate trust context.
- **KTD2:** Read the endpoint from `TYPESAFE_BASE_URL`, then `TYPESAFE_API_BASE`, then the official API; always read credentials from `TYPESAFE_API_KEY`.
- **KTD3:** Rely on `BaseSOM`'s existing `extra="forbid"` contract so old or malicious transport fields fail closed.
- **KTD4:** Keep judge serialization intact because kind/model are behavior, not secret-bearing transport configuration.

## Implementation Units

### U1. Add failing security characterization

**Goal:** Prove nested untrusted scenario JSON cannot carry TypeSafe transport configuration.

**Requirements:** R1, R2, R5

**Dependencies:** None

**Files:** `libs/giskard-checks/tests/builtin/test_som.py`, `libs/giskard-agents/tests/test_som.py`

**Approach:** Add validation coverage for a scenario containing attacker-selected `base_url` and `api_key_env`, plus direct BaseSOM validation coverage.

**Execution note:** Establish the regression failure before changing production code.

**Test scenarios:**

- A nested TypeSafe judge with either removed transport field fails validation before execution.
- A direct TypeSafe provider payload with removed transport fields fails with `extra_forbidden`.

**Verification:** The new tests fail against the vulnerable implementation for the expected reason.

### U2. Make TypeSafe transport environment-only

**Goal:** Eliminate attacker-controlled URL and credential selection while retaining local gateway support.

**Requirements:** R1, R2, R3, R4, R5

**Dependencies:** U1

**Files:** `libs/giskard-agents/src/giskard/agents/som/typesafe.py`, `libs/giskard-agents/tests/test_som.py`, `libs/giskard-checks/tests/builtin/test_som.py`

**Approach:** Remove model fields for transport overrides, resolve the endpoint only from existing TypeSafe-specific environment variables, and use the fixed TypeSafe credential variable. Update transport tests and retain the safe scenario judge round-trip test.

**Patterns to follow:** Existing environment precedence and `BaseSOM` extra-field rejection.

**Test scenarios:**

- Default configuration targets the official API.
- `TYPESAFE_BASE_URL` and `TYPESAFE_API_BASE` support local gateways with documented precedence.
- `TYPESAFE_API_KEY` supplies the bearer credential for both direct and gateway endpoints.
- Safe TypeSafe kind/model configuration survives scenario serialization and deserialization.

**Verification:** Focused agent/check tests pass and the serialized TypeSafe model contains no transport controls.

## Verification Contract

- Run focused SOM tests in `giskard-agents` and `giskard-checks` first.
- Run `make format`, `make check`, and `make test-unit PACKAGE=giskard-agents`, then `make test-unit PACKAGE=giskard-checks`.
- If `make check` fails only because its documented network-backed audit/license steps are offline, report that separately with the successful local checks.
- Review the final diff for unrelated formatting or behavior changes.

## Definition of Done

- All requirements R1-R5 are covered by passing tests.
- Untrusted transport fields fail before a request can be made.
- Environment-only gateway behavior is documented in code and tests.
- Required repository verification output is recorded.
- No abandoned experimental code or unrelated changes remain.

## Review Results

- Spec compliance: approved with no findings; R1-R5 are implemented and covered.
- Code quality/security: approved with no actionable P0-P3 findings.
- Focused SOM suites: 65 passed.
- `make format`: passed; Ruff fixed one import-order issue.
- `make test-unit PACKAGE=giskard-agents`: 194 passed, 2 skipped, 16 deselected.
- `make test-unit PACKAGE=giskard-checks`: 1054 passed, 4 skipped, 1 collection warning.
- `make check`: lint, formatting, Python compatibility, and type checking passed; the security stage failed on seven existing dependency advisories in `litellm`, `tornado`, and `urllib3`.
