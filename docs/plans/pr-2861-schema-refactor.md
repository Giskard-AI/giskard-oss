# PR 2861 schema refactor

## Scope

Keep structured output conversion in provider translators. Use Anthropic's SDK
schema transform and recursively close OpenAI/Azure object schemas while preserving
explicit map value schemas. Restore Google behavior and remove the shared profile
and mutation-policy machinery.

## Completed

- [x] Inspect serializer registrations and response-format validators.
- [x] Refactor Anthropic and OpenAI translators; fix Anthropic dict input.
- [x] Reduce tests to provider payload regressions.
- [x] Run formatting, checks, and `giskard-llm` unit tests.

## Live integration follow-up

- [x] Add a nested Anthropic structured-output functional test.
- [x] Run integration jobs on Python 3.14 while unit tests cover version breadth.
- [x] Verify workflow syntax, Python 3.14 dependency resolution, and test selection.
- [ ] Confirm the Anthropic CI job and update the draft PR.

## Review

The provider-local validators match the existing response-format conversion pattern.
PR #2862 merged and this branch was rebased onto main before the follow-up.
`make format`, `make check`, and `make test-unit PACKAGE=giskard-llm` passed.
The Anthropic selector collects the new test; Python 3.14 dry runs resolve both
scan integration dependency groups.
The new Anthropic nested test passed locally against the live API (1 passed).
