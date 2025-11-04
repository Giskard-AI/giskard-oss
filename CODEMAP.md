# CODEMAP

This codemap provides a high-level overview of the `giskard-checks` repository: purpose, architecture, modules, key types, and typical workflows. It is intended for contributors and advanced users.

### What is this library?
- Lightweight primitives to define and run checks against model interactions.
- Small, explicit, and type-safe with Pydantic v2 models.
- Async-friendly: checks can be sync or async.
- Results are immutable and easy to serialize.

### Repository layout

```
/ (repo root)
├─ pyproject.toml              # Build, tooling, dependencies
├─ README.md                   # End-user quickstart and API overview
├─ CODEMAP.md                  # This file
├─ src/giskard/checks/         # Package source (src-layout)
│  ├─ __init__.py              # Public re-exports: core, generators, testing, checks
│  ├─ core/                    # Core abstractions: Check, CheckResult, InteractionResult
│  │  ├─ __init__.py
│  │  ├─ check.py              # Check, CheckResult, CheckStatus, Metric
│  │  ├─ context.py            # Context for interaction generation
│  │  ├─ extraction.py         # Extractor, JsonPathExtractor
│  │  └─ interaction_result.py # InteractionResult[In, Out]
│  ├─ generators/              # Interaction generation system
│  │  ├─ __init__.py
│  │  ├─ base.py               # InteractionGenerator base class
│  │  ├─ static.py             # Interaction (static interactions)
│  │  └─ dynamic.py            # DynamicInteraction (callable-based)
│  ├─ checks/                  # Built-in checks and helpers
│  │  ├─ __init__.py
│  │  ├─ base.py               # BaseLLMCheck, LLMCheckResult
│  │  ├─ fn.py                 # from_fn and FnCheck
│  │  ├─ equality.py           # EqualityCheck
│  │  ├─ string_matching.py    # StringMatchingCheck
│  │  ├─ extraction_check.py   # ExtractionCheck
│  │  ├─ groundedness.py       # Groundedness check
│  │  ├─ conformity.py         # Conformity check
│  │  └─ judge.py              # LLMJudge check
│  ├─ testing/                 # TestCase, TestRunner, samples
│  │  ├─ __init__.py
│  │  ├─ testcase.py           # TestCase model + serialization helpers
│  │  └─ runner.py             # TestRunner + TestCaseResult
│  ├─ prompts/                 # Jinja2 templates for LLM checks
│  │  └─ checks/
│  │     ├─ groundedness.j2
│  │     └─ conformity.j2
│  ├─ settings.py              # Global generator settings
│  └─ utils/                   # Utility functions
│     ├─ __init__.py
│     └─ discriminated.py      # Discriminated union infrastructure
└─ tests/
   ├─ integration/             # Serialization and E2E tests
   ├─ unit/                    # Unit tests
   └─ test_utils/              # Test utilities and mocks
```

### Core concepts and types

- InteractionResult[InputT, OutputT] (in `core/interaction_result.py`)
  - Container for interaction data used internally by checks.
  - Contains inputs, outputs, and optional metadata from interaction generation.
  - This is the result of calling `InteractionGenerator.generate()`.

- InteractionGenerator (in `generators/base.py`)
  - Base class for generating interactions. Subclasses use `@InteractionGenerator.register("kind")` decorator.
  - Key method: `async generate(context: Context) -> InteractionResult[Any, Any]`.
  - Discriminated union support for polymorphic serialization and deserialization.
  - Automatic registration when classes are imported.

- Check (in `core/check.py`)
  - Base class to implement concrete checks. Subclasses use `@Check.register("kind")` decorator.
  - Key fields: `name`, `description`. `kind` is a computed field from registration.
  - Key method: `async run(interaction: InteractionResult[Any, Any]) -> CheckResult`.
  - Discriminated union support for polymorphic serialization and deserialization.
  - Automatic registration when classes are imported.

- CheckResult (in `core/check.py`)
  - Immutable result model for a single check execution.
  - Convenience constructors: `success`, `failure`, `skip`, `error`.
  - Convenience booleans: `passed`, `failed`, `errored`, `skipped`.
  - Includes `Metric` support for quantitative measurements.

- Context (in `core/context.py`)
  - Context for generating interactions containing previous interaction history.
  - Used by interaction generators to create context-aware test cases.

- Extractor and JsonPathExtractor (in `core/extraction.py`)
  - Base classes for extracting values from interactions.
  - `JsonPathExtractor` uses JSONPath expressions to extract specific fields from interaction data.

- Discriminated union infrastructure (in `utils/discriminated.py`)
  - `Discriminated`: Base class for polymorphic types with automatic `kind` field
  - `discriminated_base`: Decorator to mark base classes for discriminated unions
  - Automatic registration and deserialization using Pydantic's discriminated unions

- TestCase and TestRunner (in `testing/`)
  - `TestCase` bundles an `InteractionGenerator` and a sequence of `Check`s and exposes `await run()`.
  - `TestRunner` executes checks sequentially, measures per-check/total durations, and returns a `TestCaseResult` aggregate (immutable, with .passed/.failed/.errored/.skipped).

### Built-in checks and helpers

- FnCheck and from_fn (in `checks/fn.py`)
  - Turn a callable into a `Check`. The callable may be sync/async and return `bool` or `CheckResult`.
  - `FnCheck` is not serializable (function excluded) and is intended for programmatic/test use.

- StringMatchingCheck (in `checks/string_matching.py`)
  - KIND: `string_matching`.
  - Generic substring matcher with optional `key` (JSONPath) and `evaluation_mode`.

- ExtractionCheck (in `checks/extraction_check.py`)
  - Abstract base class for checks that extract values from interactions and evaluate them.
  - Provides common functionality for value extraction and evaluation patterns.

- BaseLLMCheck (in `checks/base.py`)
  - Abstract base class for LLM-based checks using Counterpoint.
  - Provides framework for creating checks that use Large Language Models to evaluate interactions.
  - Includes `LLMCheckResult` model for structured LLM outputs.

- Groundedness (in `checks/groundedness.py`)
  - LLM-based check for evaluating response groundedness using Counterpoint.
  - Uses Jinja2 templates for prompt generation.

- Conformity (in `checks/conformity.py`)
  - LLM-based check for evaluating response conformity using Counterpoint.
  - Uses Jinja2 templates for prompt generation.

- LLMJudge (in `checks/judge.py`)
  - Generic LLM-based judge for custom evaluation criteria.
  - Flexible check that can be configured with custom prompts.

### Interaction generators

- InteractionGenerator (in `generators/base.py`)
  - Base class for all interaction generators with discriminated union support.
  - Key method: `async generate(context: Context) -> InteractionResult[Any, Any]`.
  - Subclasses use `@InteractionGenerator.register("kind")` decorator.

- Interaction (in `generators/static.py`)
  - KIND: `static`.
  - Static interaction generator with pre-defined inputs and outputs.
  - Most common type used for testing with fixed data.

- DynamicInteraction (in `generators/dynamic.py`)
  - KIND: `dynamic`.
  - Dynamic interaction generator using callable functions.
  - Supports both sync and async callables with optional Context parameter.
  - Not serializable (function excluded) and intended for programmatic/test use.


### Serialization

The library uses standard Pydantic serialization with discriminated unions for polymorphic types.

- All models use standard Pydantic `model_dump()` and `model_validate()` methods
- Discriminated unions automatically handle polymorphic serialization using the `kind` field
- No custom serialization logic is needed

### Typical workflows

- Define an interaction generator
  - Use `Interaction` for static interactions with pre-defined inputs and outputs.
  - Use `DynamicInteraction` for dynamic interactions generated by callable functions.
  - Implement custom `InteractionGenerator` subclasses for complex generation logic.

- Author checks
  - Implement a concrete `Check` subclass with a unique `KIND`, or use `from_fn` for quick function-based checks.
  - For LLM-based checks, extend `BaseLLMCheck` and implement `get_prompt()` method.
  - Return `CheckResult` explicitly or a `bool` from `from_fn` callables.

- Run tests
  - Create a `TestCase(interaction=generator, checks=[...], name=...)` and `await tc.run()`.
  - Inspect `TestCaseResult.results`, `duration_ms`, and convenience booleans.
  - Use `await tc.assert_passed()` for assertion-based testing.

### Tooling and conventions

- Python >= 3.11 (enforced in `pyproject.toml`).
- Linting: Ruff (`E`, `W`, `I`; line length E501 ignored).
- Type checking: basedpyright (Pyright), `recommended` mode.
- Testing: pytest with `asyncio_mode = auto`.
- Development workflow: Makefile with common commands (see `make help` for full list):
  - `make setup` - Complete development setup (install deps + tools)
  - `make all` - Format, check, and test
  - `make test` - Run all tests
  - `make lint` - Run linting checks
  - `make format` - Format code with ruff
  - `make typecheck` - Run type checking with basedpyright
  - `make check` - Run all checks (lint, format, compatibility, typecheck)
  - `make ci` - Run the same checks as CI

### Environment knobs

- `GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS` (default truthy):
  - Enforces uniqueness of `Check.KIND` across registered classes; otherwise duplicates warn and last writer wins.

### Dependencies

- Runtime:
  - `pydantic>=2.11.7,<3` - Core data validation and serialization
  - `counterpoint>=0.2.3,<1` - LLM integration and workflow management
  - `jsonpath-ng>=1.7.0,<2` - JSONPath expressions for data extraction
  - `jinja2>=3.1.6,<4` - Template engine for LLM prompts
- Dev:
  - `pytest==8.4.2` - Testing framework
  - `pytest-asyncio==1.2.0` - Async test support

### Testing overview

- Integration tests under `tests/integration` cover:
  - End-to-end chat serialization using `Interaction` with `StringMatchingCheck`.
  - Structured moderation example using `Interaction`.
  - TestCase serialization/deserialization round-trips including lazy import behavior.
  - LLM-based checks using Counterpoint integration.
- Unit tests under `tests/unit` cover individual components and modules.
- Test utilities in `tests/test_utils` provide mocks and helpers for testing.

### Public API surface

Import namespaces re-exported by `giskard.checks.__init__`:

```python
from giskard.checks import core, generators, testing, checks
```

- `core` → `Check`, `CheckResult`, `CheckStatus`, `InteractionResult`
- `generators` → `InteractionGenerator`, `Interaction`, `DynamicInteraction`
- `testing` → `TestCase`, `get_runner` (via module import)
- `checks` → `from_fn`, `FnCheck`, `StringMatchingCheck`, `EqualityCheck`, `ExtractionCheck`, `Groundedness`, `Conformity`, `LLMJudge`, `BaseLLMCheck`
- `settings` → `set_default_generator`, `get_default_generator` (imported directly)

### Contributing notes

- Keep `README.md` examples in sync with APIs (especially async patterns and `uv` commands).
- Prefer absolute imports within the `giskard.checks` package.
- Ensure all public functions/classes have type hints and concise docstrings.
- Avoid adding new runtime dependencies unless necessary; prefer small, composable APIs.
