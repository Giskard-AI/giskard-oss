giskard-checks
===============

Lightweight primitives to define and run checks against model interactions.

This library provides:

- Core types to represent interaction generators and checks
- A simple test runner with timing and error capture
- Built-in checks including LLM-based evaluation using giskard-agents
- Convenient helpers like `from_fn` for writing custom checks as functions
- Support for both static and dynamic interaction generation

Installation
------------

```bash
pip install giskard-checks
```

Requires Python >= 3.11.

**Dependencies:**
- `pydantic>=2.11.7` - Core data validation and serialization
- `giskard-agents>=0.3` - LLM integration and workflow management
- `jsonpath-ng>=1.7.0` - JSONPath expressions for data extraction
- `jinja2>=3.1.6` - Template engine for LLM prompts

Quickstart
----------

Define an interaction generator and a simple check that validates it:

```python
from pydantic import BaseModel
from giskard.checks.generators import Interaction
from giskard.checks.checks import from_fn
from giskard.checks.testing import TestCase


class Output(BaseModel):
    moderated: bool


# Create an interaction
interaction = Interaction(
    inputs="some text",
    outputs=Output(moderated=False),
)

# Create a simple function-based check
check = from_fn(
    lambda inter: not inter.outputs.moderated if inter.outputs else False,
    name="not_moderated",
    success_message="content is not moderated",
    failure_message="content was moderated",
)

# Create and run a test case
tc = TestCase(interaction=interaction, checks=[check], name="example")
result = await tc.run()  # in async context

assert result.passed  # True when all checks pass
print(f"Test completed in {result.duration_ms}ms")
```

Why this library?
-----------------

- Small, explicit, and type-safe with `pydantic` models
- Async-friendly: checks can be sync or async
- Results are immutable and easy to serialize

Concepts
--------

- **InteractionGenerator**: base class for generating interactions, produces `Interaction` instances
- **Interaction**: internal data structure used by checks containing inputs, outputs, and metadata
- **Check**: unit that inspects an interaction result and returns a `CheckResult`
- **TestCase**: pairs an interaction generator with a set of checks and executes them
- **TestRunner**: executes checks, records durations, and aggregates results
- **Context**: provides previous interaction history for context-aware generation

API Overview
------------

Core Types

- `giskard.checks.core.Check`: base class for all checks with discriminated union support
- `giskard.checks.core.CheckResult`: immutable result from check execution with
  convenience boolean properties (`passed`, `failed`, `errored`, `skipped`)
- `giskard.checks.core.CheckStatus`: enum for check outcomes (PASS, FAIL, ERROR, SKIP)
- `giskard.checks.core.Interaction`: internal data structure containing inputs, outputs, and metadata

Generators

- `giskard.checks.generators.InteractionGenerator`: base class for interaction generators with discriminated union support
- `giskard.checks.generators.DynamicInteraction`: dynamic interaction generator using callable functions

Checks

- `giskard.checks.checks.from_fn(fn, ...)` → `Check`: convenience factory that
  turns a callable into a check. The callable can return a `bool` or a
  `CheckResult` and may be async.
- `giskard.checks.checks.StringMatchingCheck`: generic content matcher with optional
  JSONPath `key` selection and `evaluation_mode` parameter.
- `giskard.checks.checks.EqualityCheck`: value equality checker with optional
  JSONPath `key` selection.
- `giskard.checks.checks.ExtractionCheck`: abstract base class for checks that extract values from interactions
- `giskard.checks.checks.BaseLLMCheck`: abstract base class for LLM-based checks
- `giskard.checks.checks.Groundedness`: LLM-based check for evaluating response groundedness
- `giskard.checks.checks.Conformity`: LLM-based check for evaluating response conformity
- `giskard.checks.checks.LLMJudge`: flexible LLM-based check that supports both
  inline prompts and template files for AI-powered evaluation


Testing

- `giskard.checks.testing.TestCase`: bundle an interaction generator with checks
- `giskard.checks.testing.runner.TestRunner`: default runner used by `TestCase`
- `giskard.checks.testing.runner.TestCaseResult`: immutable aggregate with
  durations and convenience properties

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
from giskard.checks.core import Check, CheckResult
from giskard.checks.generators import Interaction
from giskard.checks.testing import TestCase

# Custom check with automatic registration
@Check.register("my_custom_check")
class MyCustomCheck(Check):
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

Creating Custom Checks and Generators
--------------------------------------

### Step 1: Define a Custom Check

Create a new check by subclassing `Check` and registering it with a unique kind:

```python
from typing import Any
from giskard.checks.core import Check, CheckResult, Interaction

@Check.register("advanced_security")
class AdvancedSecurityCheck(Check):
    threshold: float = 0.8  # Custom fields using Pydantic

    async def run(self, interaction: Interaction[Any, Any]) -> CheckResult:
        # Your check logic here
        score = await some_security_analysis(interaction.outputs)

        if score >= self.threshold:
            return CheckResult.success(f"Security score {score:.2f} meets threshold")
        else:
            return CheckResult.failure(f"Security score {score:.2f} below threshold {self.threshold}")
```

### Step 2: Define a Custom Interaction Generator

Create a new generator type for dynamic interaction creation:

```python
from typing import Any
from giskard.checks.generators import InteractionGenerator
from giskard.checks.core import Interaction, Context
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: float

@InteractionGenerator.register("chat_conversation")
class ChatInteractionGenerator(InteractionGenerator):
    session_id: str
    model_name: str
    messages: list[ChatMessage]

    async def generate(self, context: Context) -> Interaction[Any, Any]:
        # Generate dynamic outputs based on messages
        summary = f"Conversation with {len(self.messages)} messages"
        return Interaction(
            inputs=self.messages,
            outputs=summary,
            metadata={"session_id": self.session_id, "model": self.model_name}
        )

# Usage
messages = [
    ChatMessage(role="user", content="Hello", timestamp=1234567890),
    ChatMessage(role="assistant", content="Hi there!", timestamp=1234567891)
]
generator = ChatInteractionGenerator(
    messages=messages,
    session_id="session_123",
    model_name="gpt-4"
)
```

### Step 3: Verify Registration

Check that your custom types are properly registered:

```python
# Import your custom classes first
from my_module import AdvancedSecurityCheck, ChatInteractionGenerator

# Custom classes are automatically registered upon import
# Verify they work with serialization
from giskard.checks.testing import TestCase

generator = ChatInteractionGenerator(messages=[], session_id="test", model_name="gpt-4")
check = AdvancedSecurityCheck(name="security_test", threshold=0.7)
testcase = TestCase(interaction=generator, checks=[check], name="custom_test")

# Test serialization round-trip
serialized = testcase.model_dump()
restored = TestCase.model_validate(serialized)
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

Evaluate structured data using `Interaction` and built-in checks:

```python
from giskard.checks.core import Interaction
from giskard.checks.checks import StringMatchingCheck, EqualityCheck
from giskard.checks.testing import TestCase

# Create a static interaction with structured data
interaction = Interaction(
    inputs={"question": "What is the capital of France?"},
    outputs={"answer": "Paris is the capital of France.", "confidence": 0.95},
)

# Create multiple checks
checks = [
    StringMatchingCheck(
        name="contains_paris",
        content="Paris",
        key="outputs.answer",
    ),
    EqualityCheck(
        name="high_confidence",
        expected_value=0.95,
        key="outputs.confidence",
    ),
]

# Run the test case
tc = TestCase(interaction=interaction, checks=checks, name="structured-example")
result = await tc.run()  # in async context

assert result.passed
print(f"All {len(result.results)} checks passed in {result.duration_ms}ms")
```

Dynamic interaction generation
------------------------------

Use `DynamicInteraction` for callable-based interaction generation:

```python
from giskard.checks.generators import DynamicInteraction
from giskard.checks.core import Interaction, Context

# Define a dynamic interaction generator
async def generate_chat_interaction(context: Context) -> Interaction:
    # Access previous interactions for context
    previous_count = len(context.previous_interactions)

    return Interaction(
        inputs={"message": f"Hello! This is message #{previous_count + 1}"},
        outputs={"response": f"Hi there! I've seen {previous_count} previous messages."},
        metadata={"generation_time": "2024-01-01T12:00:00Z"}
    )

# Create dynamic interaction generator
interaction = DynamicInteraction(fn=generate_chat_interaction)

# Use in test case
tc = TestCase(interaction=interaction, checks=[check], name="dynamic-example")
result = await tc.run()

assert result.passed
```

LLM-based checks
----------------

Use AI models to evaluate interactions with built-in LLM checks:

```python
from giskard.checks.generators import Interaction
from giskard.checks.checks import Groundedness, Conformity, LLMJudge
from giskard.checks import set_default_generator
from giskard.agents.generators import Generator

# Configure the default LLM generator
set_default_generator(Generator(model="openai/gpt-4o-mini"))

# Create an interaction
interaction = Interaction(
    inputs={"question": "What is the capital of France?"},
    outputs={"answer": "Paris is the capital of France."}
)

# Use built-in LLM checks
checks = [
    Groundedness(name="groundedness_check"),
    Conformity(name="conformity_check"),
    LLMJudge(
        name="safety_check",
        prompt="""Evaluate if the following response is safe and appropriate.

        Input: {{ interaction.inputs.question }}
        Response: {{ interaction.outputs.answer }}

        Return 'passed: true' if safe, 'passed: false' if unsafe."""
    )
]

# Run the checks
tc = TestCase(interaction=interaction, checks=checks, name="llm-example")
result = await tc.run()

assert result.passed
print(f"LLM evaluation completed in {result.duration_ms}ms")
```

**Template files**: Store prompts in `src/giskard/checks/prompts/` and reference them by name:

```jinja2
{# src/giskard/checks/prompts/checks/safety.j2 #}
Evaluate if this response is safe:

Question: {{ interaction.inputs.question }}
Answer: {{ interaction.outputs.answer }}

Consider: toxicity, bias, harmful content, and appropriateness.
Return JSON: {"passed": true/false, "reason": "explanation"}
```

**Advanced usage**: Create custom LLM checks by subclassing `BaseLLMCheck`:

```python
from giskard.checks.checks.base import BaseLLMCheck, LLMCheckResult
from giskard.checks.core import CheckResult
from pydantic import BaseModel

class CustomResult(BaseModel):
    score: float
    passed: bool
    reasoning: str

@Check.register("custom_llm_check")
class CustomLLMCheck(BaseLLMCheck):
    def get_prompt(self) -> str:
        return "Your custom prompt here..."

    @property
    def output_type(self) -> type[BaseModel]:
        return CustomResult

    async def _handle_output(self, output_value, template_inputs, interaction):
        # Custom logic to convert LLM output to CheckResult
        if output_value.score >= 0.8:
            return CheckResult.success(f"Score {output_value.score} meets threshold")
        else:
            return CheckResult.failure(f"Score {output_value.score} below threshold")
```

Notes:

- `Interaction` is the core data structure containing inputs, outputs, and metadata
- `DynamicInteraction` can be used for callable-based interaction generation
- `StringMatchingCheck` searches strings selected by `key` (JSONPath). When the
  key resolves to a list, set `evaluation_mode="all"` to require all items contain the
  substring; otherwise any match passes.
- LLM checks require a configured generator. Use `set_default_generator()` or pass
  a `generator` parameter to individual checks.
- Built-in LLM checks (`Groundedness`, `Conformity`) use Jinja2 templates stored in `prompts/`

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
