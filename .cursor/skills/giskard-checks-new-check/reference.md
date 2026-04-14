# Reference — check variants in giskard-checks

## JSONPath enforcement and helpers

- **Source**: `libs/giskard-checks/src/giskard/checks/core/extraction.py` — `JSONPathStr`, `resolve`, `provided_or_resolve`, `NoMatch`.
- **Enforcement**: `libs/giskard-checks/tests/core/test_jsonpath_enforcement.py` — field names `key` or `*_key` must carry the `JSONPathStr` marker (including inside unions with `None` or `NotProvided`).
- **`NotProvided` pattern** (`ComparisonCheck`): `expected_value_key: JSONPathStr | NotProvided` with `default=NOT_PROVIDED`; in `run`, use `provided_or_resolve(trace, key=provide_not_none(self.expected_value_key), value=self.expected_value)` so a literal `expected_value` skips the path.
- **`None` optional path** (`StringMatching.keyword_key`): `JSONPathStr | None` with a validator that requires exactly one of inline value or key—do not use a bare `str` annotation for any `*_key` field.
- **Multi-match**: `resolve` returns a list when the expression yields multiple matches or the parsed path is list-like (`Slice`, `Union`, etc.); single-match paths return one value or `NoMatch`.

## Non-LLM `Check`

- See `libs/giskard-checks/src/giskard/checks/builtin/` for patterns: comparison checks, `FnCheck`, `StringMatching` / `RegexMatching`, `SemanticSimilarity`.
- Subclass `Check[InputType, OutputType, TraceType]` when generics matter; many builtins use `Trace[dict[str, str], dict[str, str]]` or similar.
- Prefer returning `CheckResult` with clear `message` and structured `details` for debugging.

## `BaseLLMCheck` (judges)

- Defined in `libs/giskard-checks/src/giskard/checks/judges/base.py`.
- Implement `get_prompt()` returning `str`, `Message`, `MessageTemplate`, or `TemplateReference`.
- Default structured output model is `LLMCheckResult` (`passed`, `reason`); override `output_type` if using a different Pydantic model for the LLM response.
- Uses `WithGeneratorMixin` — generator defaults come from library settings (`set_default_generator` / `get_default_generator` in public API).
- Examples: `Groundedness`, `Conformity`, `LLMJudge` in the same `judges/` package.

## Export surfaces

Typical touch points for a **public** check:

1. `builtin/__init__.py` or `judges/__init__.py` — import and re-export; update `__all__`.
2. `src/giskard/checks/__init__.py` — import from submodule and add to `__all__` alongside other checks.

Internal-only types are rare; if you add one, skip top-level exports but still register the kind and import the module where deserialization must work.
