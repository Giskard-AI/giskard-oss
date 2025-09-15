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
├─ src/giskard_checks/         # Package source (src-layout)
│  ├─ __init__.py              # Public re-exports: core, interactions, testing, checks
│  ├─ core/                    # Core abstractions: Check, CheckResult, enums
│  │  ├─ __init__.py
│  │  ├─ check.py
│  │  ├─ extraction.py         # Extractor, JsonPathExtractor
│  │  └─ interactions.py
│  ├─ interactions/            # Interaction specializations
│  │  ├─ __init__.py
│  │  └─ structured.py         # StructuredInteraction[In, Out]
│  ├─ checks/                  # Built-in checks and helpers
│  │  ├─ __init__.py
│  │  ├─ fn.py                 # from_fn and FnCheck
│  │  ├─ equality.py           # EqualityCheck
│  │  ├─ string_matching.py    # StringMatchingCheck
│  │  └─ extraction_check.py   # ExtractionCheck
│  └─ testing/                 # TestCase, TestRunner, samples
│     ├─ __init__.py
│     ├─ testcase.py           # TestCase model + serialization helpers
│     ├─ runner.py             # TestRunner + TestCaseResult
│     └─ _samples/             # Example custom checks/interactions
└─ tests/
   └─ integration/             # Serialization and E2E tests
```

### Core concepts and types

- Interaction[InputT, OutputT] (in `core/interactions.py`)
  - Container for an input, optional output, and optional metadata.
  - Specialized as `StructuredInteraction[In, Out]`.

- Check[InteractionT] (in `core/check.py`)
  - Base class to implement concrete checks. Subclasses must define a class-level `KIND: str`.
  - Key fields: `name`, `description`. `kind` is a computed field derived from `KIND`.
  - Global registry keyed by `KIND` to support deserialization and uniqueness validation.

- CheckResult (in `core/check.py`)
  - Immutable result model for a single check execution.
  - Convenience constructors: `success`, `failure`, `skip`, `error`.
  - Convenience booleans: `passed`, `failed`, `errored`, `skipped`.

- Extractor and JsonPathExtractor (in `core/extraction.py`)
  - Base classes for extracting values from interactions.
  - `JsonPathExtractor` uses JSONPath expressions to extract specific fields from interaction data.

- TestCase and TestRunner (in `testing/`)
  - `TestCase` bundles an `Interaction` and a sequence of `Check`s and exposes `await run()`.
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

### Interaction specializations

- StructuredInteraction[In, Out] (in `interactions/structured.py`)
  - Typed interaction for structured payloads.


### Serialization model

- Checks
  - Each `Check` exposes a computed `kind` used during serialization.
  - Deserialization uses the global `KIND` registry and may lazily import classes when `__type__` is provided.

- TestCase
  - `serialize()` embeds the fully-qualified class path of the `Interaction` in `interaction.__type__` and its data in `interaction.data`.
  - Each check payload includes `__type__` for lazy import during deserialization.
  - `deserialize()` reconstructs `Interaction` and checks using the above information.

### Typical workflows

- Define an interaction
  - Use `StructuredInteraction` for typed input/output models (Pydantic `BaseModel` recommended) including chat transcripts.

- Author checks
  - Implement a concrete `Check` subclass with a unique `KIND`, or use `from_fn` for quick function-based checks.
  - Return `CheckResult` explicitly or a `bool` from `from_fn` callables.

- Run tests
  - Create a `TestCase(interaction=..., checks=[...], name=...)` and `await tc.run()`.
  - Inspect `TestCaseResult.results`, `duration_ms`, and convenience booleans.

### Tooling and conventions

- Python >= 3.11 (enforced in `pyproject.toml`).
- Linting: Ruff (`E`, `W`, `I`; line length E501 ignored).
- Type checking: Pyright, `recommended` mode.
- Testing: pytest with `asyncio_mode = auto`.
- Development workflow: Makefile with common commands (see `make help` for full list):
  - `make setup` - Complete development setup (install deps + tools)
  - `make all` - Format, check, and test
  - `make test` - Run all tests
  - `make lint` - Run linting checks
  - `make format` - Format code with ruff
  - `make check` - Run all checks (lint, format, compatibility)
  - `make ci` - Run the same checks as CI

### Environment knobs

- `GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS` (default truthy):
  - Enforces uniqueness of `Check.KIND` across registered classes; otherwise duplicates warn and last writer wins.

### Dependencies

- Runtime:
  - `pydantic~=2.11`
- Dev:
  - `pytest`, `pytest-asyncio`

### Testing overview

- Integration tests under `tests/integration` cover:
  - End-to-end chat serialization using `StructuredInteraction` with `StringMatchingCheck`.
  - Structured moderation example using `StructuredInteraction`.
  - TestCase serialization/deserialization round-trips including lazy import behavior.

### Public API surface

Import namespaces re-exported by `giskard_checks.__init__`:

```python
from giskard_checks import core, interactions, testing, checks
```

- `core` → `Check`, `CheckResult`, `Interaction`
- `interactions` → `StructuredInteraction`
- `testing` → `TestCase`, `runner` (via module import)
- `checks` → `from_fn`, `FnCheck`, `StringMatchingCheck`, `EqualityCheck`, `ExtractionCheck`

### Contributing notes

- Keep `README.md` examples in sync with APIs (especially async patterns and `uv` commands).
- Prefer absolute imports within the `giskard_checks` package.
- Ensure all public functions/classes have type hints and concise docstrings.
- Avoid adding new runtime dependencies unless necessary; prefer small, composable APIs.
