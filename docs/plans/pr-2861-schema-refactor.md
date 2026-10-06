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

## Review

The provider-local validators match the existing response-format conversion pattern.
`make format` passed. `make check` passed lint, format, and typecheck, then failed
pip-audit on fsspec and multidict; PR #2862 contains that dependency fix and is
still open. `make test-unit PACKAGE=giskard-llm`: 235 passed, 7 skipped.
