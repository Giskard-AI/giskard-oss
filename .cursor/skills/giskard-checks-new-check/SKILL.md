---
name: giskard-checks-new-check
description: Guides implementation of new or changed eval checks in the giskard-checks package—Check subclasses, discriminated kind registration, public exports, and tests. Use when adding or modifying built-in checks, LLM-based judges, @Check.register kinds, serialization of TestCase or Scenario, or when editing files under libs/giskard-checks/src/giskard/checks related to checks.
---

# giskard-checks — new check

## Scope

Library code lives under `libs/giskard-checks/`. Follow existing patterns; use absolute imports inside `giskard.checks`. Python 3.12+, Pydantic v2.

Read project rules when relevant: `libs/giskard-checks/.cursor/rules/02-project.mdc`, `03-development.mdc`.

## Classify the check

| Kind | Module | Base class |
|------|--------|------------|
| Deterministic (string, numeric, custom logic on trace) | `src/giskard/checks/builtin/` | `Check` (often shared helpers in existing builtins) |
| Uses LLM to score or classify | `src/giskard/checks/judges/` | `BaseLLMCheck` |

For one-off or experimental logic, users can use `from_fn` / `FnCheck` without a new class—only add a new type when it is a reusable, serializable primitive.

## JSONPath fields (`key` and `*_key`)

### Naming and typing (required)

Any `Check` field named `key` or whose name ends with `_key` is a JSONPath into the trace. It **must** be annotated with `JSONPathStr`, or `JSONPathStr | None`, or `JSONPathStr | NotProvided` (from `..core.extraction` and `giskard.core`). Otherwise `tests/core/test_jsonpath_enforcement.py` fails.

Inside the library, import from `..core.extraction` (module `giskard.checks.core.extraction`).

### Path syntax

- Every path must start with `trace.` (validated at model construction via `jsonpath_ng`).
- Resolution runs against `{"trace": trace.model_dump()}`. Prefer `trace.last.outputs`, `trace.last.inputs`, `trace.last.metadata.*` over brittle indices; `trace.last` is the usual default for “current turn” data.

### Reading values

- **`resolve(trace, key)`** — Use when the value always comes from the trace (e.g. comparison `key`).
- **`provided_or_resolve(trace, key=..., value=...)`** — Use when the user may pass an inline value **or** fall back to a path (e.g. `answer` + `answer_key`). If `value` is provided (`provide_not_none` helps for optional fields), that wins; otherwise the path is evaluated.

Optional key fields that are only sometimes used: type as `JSONPathStr | None` or `JSONPathStr | NotProvided`, pass through `provide_not_none` when calling `provided_or_resolve`, and validate combinations with a `@model_validator` if only one of (inline, `*_key`) may be set (see `StringMatching` / `ComparisonCheck`).

### Failures and types

- Missing matches become **`NoMatch`**. Check with `isinstance(x, NoMatch)` and return `CheckResult.failure` with a message that names the field/key.
- Some paths return a **list** (multiple matches or list-producing JSONPath). See `resolve` in `core/extraction.py` if the check must treat collections differently.

### Defaults

Use `Field(default="trace.last.outputs", ...)` (or another sensible default) so typical scenarios need no extra configuration.

More detail: [reference.md](reference.md).

## Implementation checklist

1. **Kind string** — Pick a unique snake_case discriminator. Search the repo for `@Check.register("` to avoid duplicates.

2. **Class** — Subclass `Check[...]` or `BaseLLMCheck[...]` from `..core.check` / `..judges.base`. Use Pydantic `Field` for config. For non-LLM checks, implement `async def run(self, trace: TraceType) -> CheckResult` (use `CheckResult.success` / `CheckResult.failure`; put extra data in `details=` when useful).

3. **Registration** — Decorate with `@Check.register("your_kind")` on the class definition.

4. **Wire imports so the kind is registered** — Discriminated unions only know subclasses that were imported. Add an import of the new module to the appropriate package `__init__.py` (e.g. `builtin/__init__.py` re-exports; `judges/__init__.py` for judges). Update `src/giskard/checks/__init__.py` exports and `__all__` if the type is public API.

5. **JSONPath fields** — Follow the section above for every `key` / `*_key`; run tests so `test_jsonpath_enforcement` passes.

6. **Tests** — Add `tests/builtin/test_<feature>.py` or extend an existing file (e.g. `test_judge.py`). Mirror package layout. Use `pytest` and async tests where `run` is async. Cover pass, fail, and edge cases (missing values, wrong types).

7. **Validate** — From the **repository root** (this monorepo uses the root `Makefile`):

   - `make test-unit PACKAGE=giskard-checks` — unit tests for that lib
   - `make check` — lint, format, compat, typecheck, etc.

   Use `make test PACKAGE=giskard-checks` if changes must coexist with functional tests.

   `make test-unit PACKAGE=giskard-checks` includes `tests/core/test_jsonpath_enforcement.py`.

## Serialization note

After `model_dump()`, `model_validate()` needs every custom `Check` subclass already imported. If deserialization tests fail with an unknown kind, ensure the defining module is imported on the test (or via package `__init__`) before validation.

## LLM checks

For `BaseLLMCheck`, implement `get_prompt` and respect `output_type` / structured parsing as in existing judges. Details: [reference.md](reference.md).
