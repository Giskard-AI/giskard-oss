giskard-checks
===============

Lightweight primitives to define and run checks against model interactions.

This library provides:

- Core types to represent interactions and checks
- A simple test runner with timing and error capture
- A tiny, convenient check helper `from_fn` for writing checks as functions

Installation
------------

```bash
pip install giskard-checks
```

Requires Python >= 3.11.

Quickstart
----------

Define an interaction and a simple check that validates it:

```python
from pydantic import BaseModel
from giskard_checks.interactions import StructuredInteraction
from giskard_checks.checks import from_fn
from giskard_checks.testing import TestCase


class Output(BaseModel):
    moderated: bool


interaction = StructuredInteraction[str, Output](
    input="some text",
    output=Output(moderated=False),
)

check = from_fn(
    lambda inter: not inter.output.moderated if inter.output else False,
    name="not_moderated",
    success_message="content is not moderated",
    failure_message="content was moderated",
)

tc = TestCase(interaction=interaction, checks=[check], name="example")
result = await tc.run()  # in async context

assert result.passed  # True when all checks pass
```

Why this library?
-----------------

- Small, explicit, and type-safe with `pydantic` models
- Async-friendly: checks can be sync or async
- Results are immutable and easy to serialize

Concepts
--------

- Interaction: container for the input, optional output, and metadata.
- Check: unit that inspects an interaction and returns a `CheckResult`.
- TestCase: pairs an interaction with a set of checks and executes them.
- TestRunner: executes checks, records durations, and aggregates results.

API Overview
------------

Core

- `giskard_checks.core.Interaction[InputT, OutputT]`: generic interaction model.
- `giskard_checks.core.Check[InteractionT]`: base class for checks.
- `giskard_checks.core.CheckResult`: immutable result of a check execution with
  convenience boolean properties (`passed`, `failed`, `errored`, `skipped`).

Checks

- `giskard_checks.checks.from_fn(fn, ...)` → `Check`: convenience factory that
  turns a callable into a check. The callable can return a `bool` or a
  `CheckResult` and may be async.
 - `giskard_checks.checks.StringMatchingCheck`: generic content matcher with optional
   JSONPath `key` selection and optional `match_all`.

Interactions

- `giskard_checks.interactions.StructuredInteraction[In, Out]`: specialization
  of `Interaction` for structured inputs/outputs.
 - `giskard_checks.interactions.ChatInteraction`: specialization for chat-style
  interactions using `counterpoint.Message` as items.

Testing

- `giskard_checks.testing.TestCase`: bundle an interaction with checks.
- `giskard_checks.testing.runner.TestRunner`: default runner used by `TestCase`.
- `giskard_checks.testing.runner.TestCaseResult`: immutable aggregate with
  durations and convenience properties.

Usage Notes
-----------

- Define your own `Check` subclasses with a unique class-level `KIND` string.
- You can customize result messages and attach additional context in `details`
  via `CheckResult` or by returning `bool` from `from_fn`.
- Environment variable `GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS` controls
  whether duplicate `KIND`s raise (default: enabled).

Chat quickstart
---------------

Evaluate chat transcripts using `ChatInteraction` and the built-in
`StringMatchingCheck`:

```python
from counterpoint import Message
from giskard_checks.interactions import ChatInteraction
from giskard_checks.checks import StringMatchingCheck
from giskard_checks.testing import TestCase

interaction = ChatInteraction(
    input=[Message(role="user", content="Say hello")],
    output=[Message(role="assistant", content="Hello world!")],
)

checks = [
    StringMatchingCheck(
        name="contains_hello",
        content="Hello",
        key="output[*].content",
    ),
]

tc = TestCase(interaction=interaction, checks=checks, name="chat-example")
result = await tc.run()  # in async context

assert result.passed
```

Notes:

- `ChatInteraction` uses `counterpoint.Message` for messages; the `counterpoint`
  package is declared as a dependency.
- `StringMatchingCheck` searches strings selected by `key` (JSONPath). When the
  key resolves to a list, set `match_all=True` to require all items contain the
  substring; otherwise any match passes.

Development
-----------

Run the test suite:

```bash
uv run pytest -q
```

Type checking:

```bash
uv run basedpyright
```
