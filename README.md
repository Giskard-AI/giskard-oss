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
from giskard_checks.core.interactions import Interaction
from giskard_checks.checks import from_fn
from giskard_checks.testing import TestCase


class Output(BaseModel):
    moderated: bool


interaction = Interaction[str, Output](
    inputs="some text",
    outputs=Output(moderated=False),
)

check = from_fn(
    lambda inter: not inter.outputs.moderated if inter.output else False,
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
- `giskard_checks.core.Check`: base class for checks.
- `giskard_checks.core.CheckResult`: immutable result of a check execution with
  convenience boolean properties (`passed`, `failed`, `errored`, `skipped`).

Core Types

- `giskard_checks.core.Check`: base class for all checks with discriminated union support
- `giskard_checks.core.CheckResult`: immutable result from check execution
- `giskard_checks.core.CheckStatus`: enum for check outcomes (PASS, FAIL, ERROR, SKIP)
- `giskard_checks.core.Interaction`: base class for all interactions with discriminated union support

Checks

- `giskard_checks.checks.from_fn(fn, ...)` → `Check`: convenience factory that
  turns a callable into a check. The callable can return a `bool` or a
  `CheckResult` and may be async.
 - `giskard_checks.checks.StringMatchingCheck`: generic content matcher with optional
   JSONPath `key` selection and `evaluation_mode` parameter.
 - `giskard_checks.checks.EqualityCheck`: value equality checker with optional
   JSONPath `key` selection.

Interactions

- `giskard_checks.core.interactions.Interaction[In, Out]`: base interaction
  for structured inputs/outputs with full type safety and serialization support.

Testing

- `giskard_checks.testing.TestCase`: bundle an interaction with checks.
- `giskard_checks.testing.runner.TestRunner`: default runner used by `TestCase`.
- `giskard_checks.testing.runner.TestCaseResult`: immutable aggregate with
  durations and convenience properties.

Usage Notes
-----------

- Define your own `Check` subclasses with a unique class-level `KIND` string.
- All custom checks and interactions are automatically registered when their classes are imported.
- Use `model_dump()` and `model_validate()` methods for reliable serialization (replaces `to_dict`/`from_dict`).
- You can customize result messages and attach additional context in `details`
  via `CheckResult` or by returning `bool` from `from_fn`.
- Environment variable `GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS` controls
  whether duplicate `KIND`s raise (default: enabled).

Serialization
-------------

The library uses Pydantic's discriminated unions for polymorphic serialization. Classes are automatically registered when imported:

```python
from giskard_checks.core import Check, Interaction
from giskard_checks.testing import TestCase

# Custom check with automatic registration
@Check.register("my_custom_check")
class MyCustomCheck(Check[Interaction[str, str]]):
    async def run(self, interaction):
        return CheckResult.success("Check passed")

# Serialize and deserialize test cases
interaction = Interaction(inputs="test", outputs="result")
check = MyCustomCheck(name="test")
testcase = TestCase(interaction=interaction, checks=[check], name="example")

# Serialize to dict
serialized = testcase.model_dump()

# Deserialize back (requires classes to be imported)
restored = TestCase.model_validate(serialized)
```

**Important**: For deserialization to work, all custom classes must be imported before calling `model_validate()`. The registry only knows about classes that have been loaded into memory.

Creating Custom Checks and Interactions
----------------------------------------

### Step 1: Define a Custom Check

Create a new check by subclassing `Check` and defining a unique `KIND`:

```python
from giskard_checks.core import Check, CheckResult, Interaction

class AdvancedSecurityCheck(Check):
    KIND = "advanced_security"  # Must be unique across all checks

    threshold: float = 0.8  # Custom fields using Pydantic

    async def run(self, interaction: Interaction[Any, Any]) -> CheckResult:
        # Your check logic here
        score = await some_security_analysis(interaction.output)

        if score >= self.threshold:
            return self.success(f"Security score {score:.2f} meets threshold")
        else:
            return self.failure(f"Security score {score:.2f} below threshold {self.threshold}")
```

### Step 2: Define a Custom Interaction

Create a new interaction type for specialized data:

```python
from giskard_checks.core import Interaction
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: float

class ChatInteraction(Interaction[list[ChatMessage], str]):
    KIND = "chat_conversation"  # Must be unique across all interactions

    session_id: str
    model_name: str

# Usage
messages = [
    ChatMessage(role="user", content="Hello", timestamp=1234567890),
    ChatMessage(role="assistant", content="Hi there!", timestamp=1234567891)
]
interaction = ChatInteraction(
    inputs=messages,
    outputs="Conversation summary",
    session_id="session_123",
    model_name="gpt-4"
)
```

### Step 3: Verify Registration

Check that your custom types are properly registered:

```python
from giskard_checks.core import list_registered_check_kinds, list_registered_interaction_kinds

# Import your custom classes first
from my_module import AdvancedSecurityCheck, ChatInteraction

# Verify registration
print("Registered check kinds:", list_registered_check_kinds())
print("Registered interaction kinds:", list_registered_interaction_kinds())

# Should include 'advanced_security' and 'chat_conversation'
```

### Step 4: Test Serialization

Verify that serialization and deserialization work correctly:

```python
from giskard_checks.testing import TestCase

# Create test case with custom types
check = AdvancedSecurityCheck(name="security_test", threshold=0.7)
testcase = TestCase(
    interaction=interaction,
    checks=[check],
    name="custom_test"
)

# Test serialization round-trip
serialized = testcase.model_dump()
restored = TestCase.model_validate(serialized)

# Verify types are preserved
assert isinstance(restored.interaction, ChatInteraction)
assert isinstance(restored.checks[0], AdvancedSecurityCheck)
assert restored.checks[0].threshold == 0.7
```

Troubleshooting Serialization Issues
------------------------------------

### Common Errors and Solutions

**ValidationError**: "Kind is not provided for Check"
- **Cause**: Custom class not imported before deserialization
- **Solution**: Import all custom classes before calling `model_validate()`
```python
# Import before deserializing
from my_module import MyCustomCheck
restored = TestCase.model_validate(data)
```

**DuplicateKindError**: "Duplicate kind 'my_check' detected"
- **Cause**: Multiple classes define the same `KIND` value
- **Solution**: Ensure each `KIND` is unique across your codebase
```python
# Bad - duplicate KIND
class CheckA(Check): KIND = "my_check"
class CheckB(Check): KIND = "my_check"  # Error!

# Good - unique KINDs
class CheckA(Check): KIND = "check_a"
class CheckB(Check): KIND = "check_b"
```

**Missing registration**
- **Cause**: Subclass not registered with a decorator
- **Solution**: Use the `@Check.register()` decorator
```python
@Check.register("my_check")
class MyCheck(Check):
    pass
```

**Import order issues in tests**
- **Cause**: Tests call `model_validate()` before importing custom classes
- **Solution**: Import custom modules in test setup
```python
import pytest
from my_module import MyCustomCheck, MyCustomInteraction  # Import first

def test_custom_serialization():
    # Now deserialization will work
    restored = TestCase.model_validate(serialized_data)
```

### Environment Variables

- `GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS=1` (default): Raises `DuplicateKindError` on conflicts
- `GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS=0`: Warns and allows last-defined class to win

Structured data quickstart
---------------------------

Evaluate structured data using `Interaction` and the built-in
`StringMatchingCheck`:

```python
from giskard_checks.core.interactions import Interaction
from giskard_checks.checks import StringMatchingCheck
from giskard_checks.testing import TestCase

interaction = Interaction(
    inputs={"question": "What is the capital of France?"},
    outputs={"answer": "Paris is the capital of France."},
)

checks = [
    StringMatchingCheck(
        name="contains_paris",
        content="Paris",
        key="outputs.answer",
    ),
]

tc = TestCase(interaction=interaction, checks=checks, name="structured-example")
result = await tc.run()  # in async context

assert result.passed
```

Notes:

- `Interaction` is the base interaction type for all data with full type safety.
- `StringMatchingCheck` searches strings selected by `key` (JSONPath). When the
  key resolves to a list, set `evaluation_mode="all"` to require all items contain the
  substring; otherwise any match passes.

Development
-----------

This project uses a Makefile for common development tasks. Run `make help` to see all available commands.

Quick start:

```bash
make setup    # Complete development setup (install deps + tools)
make all      # Format, check, and test
```

Common commands:

```bash
make test     # Run all tests
make lint     # Run linting checks
make format   # Format code with ruff
make typecheck # Run type checking with basedpyright
make check    # Run all checks (lint, format, compatibility, typecheck)
make ci       # Run the same checks as CI
```

For more details, see the [Makefile](Makefile) or run `make help`.
