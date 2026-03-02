# CODEMAP

This document is an orientation guide for contributors: repository layout, key
modules, core abstractions, and expected workflows.

## What is this library?

- Lightweight primitives to define and run checks against model interactions.
- Type-safe, async-friendly, and immutable by design via Pydantic v2.
- Built-in extraction utilities and LLM-based checks powered by `giskard-agents`.

## Repository layout

```
/ (repo root)
├─ pyproject.toml              # Build, tooling, dependencies
├─ README.md                   # End-user quickstart & API overview
├─ CODEMAP.md                  # This file
├─ Makefile                    # Canonical dev workflow (lint, typecheck, tests)
├─ src/giskard/checks/
│  ├─ __init__.py              # Public re-exports: all core classes, builtin/LLM-based checks, and settings helpers
│  ├─ builtin/                 # Built-in Check implementations (fn, equality, LLM, etc.)
│  ├─ core/                    # Core abstractions: Interaction, Check, Scenario, Trace, results, extraction
│  ├─ generators/              # User-facing generators (e.g., UserSimulator) and related helpers
│  ├─ judges/                  # LLM-based Check implementations (Groundedness, Conformity, etc.)
│  ├─ prompts/                 # Jinja templates used by generators and judges (LLM prompts)
│  ├─ scenarios/               # Runner, TestCase model, and helper utilities
│  ├─ settings.py              # Global generator configuration for LLM checks
│  ├─ testing/                 # Testing helpers (spies, runners) for checks and scenarios
│  └─ utils/                   # Shared utilities: value providers, normalization, generator helpers, parameter injection
└─ tests/
   ├─ builtin/                 # Tests for builtin and LLM-based checks
   ├─ conftest.py              # Shared pytest fixtures and configuration for tests
   ├─ core/                    # Unit tests for Check/Scenario primitives
   ├─ generators/              # Tests for user simulator and generator helpers
   ├─ integration/             # End-to-end scenarios exercising multiple components together
   ├─ scenarios/               # TestCase + runner scenarios
   ├─ trace/                   # Trace/interaction behavior
   └─ utils/                   # Tests for utility helpers (normalization, etc.)
```

## Core concepts & files

### Trace & Interactions (`core/trace.py`)
- `Interaction`: immutable payload with `inputs`, `outputs`, `metadata`.
- `Trace`: ordered list of interactions; passed to every `Check`.

### Scenario components (`core/scenario.py`, `core/interaction.py`, `core/check.py`)
- `ScenarioComponent`: discriminated base for anything that can be executed in a scenario
  (either an `Interact` or a `Check`).
- `Scenario`: ordered sequence of components with a shared `Trace`. Components execute
  sequentially, stopping at the first failing check. Supports custom trace types.
- `InteractionSpec`: base class for specs that emit `Interaction` objects via
  the `generate()` async generator method. Each yielded interaction receives the updated
  trace via `generator.asend()`.
- `Interact` (`interaction/interact.py`): default implementation accepting
  static values, callables, or generators for both inputs and outputs. Supports
  multi-turn interactions through generators.
- `Check`: base class for executable validations; subclasses return `CheckResult`.
- Registration via `@Check.register("kind")` and `@InteractionSpec.register("kind")`
  enables polymorphic serialization.

### Results (`core/result.py`)
- `CheckStatus`, `CheckResult`, `Metric`: immutable check outcomes with helpers.
- `ScenarioResult`: aggregated results + final trace for a single scenario execution.
- `TestCaseResult`: multi-run summary with helper predicates (`passed`, `failed`, etc.)
  and `format_failures()` / `assert_passed()` helpers.

### Extraction (`core/extraction.py`)
- `Extractor` base class plus `JsonPathExtractor`.
- `resolve()` helper to evaluate JSONPath expressions against a trace.
- Used heavily by builtin checks via `ExtractionCheck`.

### Scenario runner (`scenarios/runner.py`)
- `ScenarioRunner`: orchestrates sequences of `ScenarioComponent`s using async
  generators and shared traces. Processes components sequentially:
  - `InteractionSpec` components: call `generate()` to yield interactions, send
    updated trace back via `asend()`
  - `Check` components: call `run()` to validate trace, stop on failure/error
- Adds duration metrics, converts exceptions into `CheckResult.error`, and stops
  on first failure.
- `TraceBuilder`: helper class for incrementally building trace instances.
- `_default_runner` singleton accessible via `get_runner()`.

### Test cases (`core/testcase.py`)
- `TestCase`: wraps a single interaction spec + a list of checks.
- `run(max_runs=1)` delegates to the runner and returns `TestCaseResult`.
- `assert_passed()` helper runs + asserts success.

### Built-in checks (`builtin/`)
- `from_fn`, `FnCheck`: wrap arbitrary callables (sync/async) that receive a `Trace`.
- Extraction-based checks: `StringMatching`, `Equality`, `ExtractionCheck`.
- LLM checks: `BaseLLMCheck`, `LLMJudge`, `Groundedness`, `Conformity`.
  These integrate with `giskard-agents` via `TemplateReference` and respect
  global/default generators configured through `settings.py`.

### Settings (`settings.py`)
- `set_default_generator` / `get_default_generator`: configure the generator used by
  LLM checks when none is supplied explicitly.

### Tests (`tests/`)
- `tests/core`: covers `Check`, `CheckResult`, extraction utilities, etc.
- `tests/scenarios`: heavy coverage of `TestCase`, runner edges, scenario behavior,
  and fixtures for async execution.
- `tests/trace`: trace/interaction specification behavior and serialization tests.

## Typical workflows

1. **Describe interactions** using `Interact` (static payloads, callables, or
   generators that can themselves depend on the current `Trace`).
2. **Author checks** by subclassing `Check` or using `from_fn`. Access the
   current output via `trace.interactions[-1]`.
3. **Bundle scenarios**:
   - `Scenario(sequence=[...])`: compose multiple interactions and checks in order
   - `await scenario.run()`
4. **Inspect results** via `ScenarioResult`:
   - `result.passed`, `result.failed`, `result.errored` convenience booleans
   - `result.check_results`: list of all check results
   - `result.final_trace`: final trace state after execution
   - `result.duration_ms`: execution time
   - `result.assert_passed()`: raise AssertionError with formatted failures

## Tooling & conventions

- Python >= 3.12 (enforced in `pyproject.toml`).
- Ruff for linting (`E`, `W`, `I`; `E501` ignored), basedpyright for typing, pytest-asyncio.
- Use Makefile commands (`make test`, `make lint`, `make ci`, etc.) instead of
  invoking tools directly.
- Prefer absolute imports within `giskard.checks`.
- Keep docstrings concise but informative; README and this CODEMAP must reflect API reality.

## Public API surface

All public classes and functions are exported from the main `giskard.checks` package:

```python
from giskard.checks import (
    # Modules
    builtin, judges

    # Core classes
    Check, CheckResult, CheckStatus, Metric,
    Scenario, ScenarioResult,
    TestCase, TestCaseResult,
    Trace, Interact,
    Interaction, InteractionSpec,

    # Builtin and LLM-based checks
    BaseLLMCheck, LLMCheckResult,
    Conformity, Equals, NotEquals,
    LesserThan, GreaterThan,
    LesserThanEquals, GreaterEquals,
    FnCheck, from_fn,
    Groundedness, LLMJudge,
    SemanticSimilarity,
    StringMatching, RegexMatching,

    # Generators
    UserSimulator,

    # Testing
    WithSpy, TestCaseRunner,

    # Scenarios
    ScenarioBuilder, scenario, ScenarioRunner,

    # Settings
    set_default_generator, get_default_generator,
)
```

- All core types, builtin and LLM-based checks, and utilities are available directly from `giskard.checks`.
- The `builtin` and `judges` modules are still accessible for accessing the submodule directly if needed.

## Environment knobs

- `GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS` (default truthy):
  raises when duplicate `KIND`s are registered; set to `0` to allow last-one-wins (with warning).

## Contributing notes

- Keep README/CODEMAP in sync with code changes—especially when reorganizing modules.
- Any new `ScenarioComponent` must opt into discriminated registration via the decorator.
- Favor immutable data structures (`frozen=True`) when extending Trace/Interaction models.
- Avoid introducing new runtime dependencies unless necessary; prefer composable helpers.
