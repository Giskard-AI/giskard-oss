giskard-checks
===============

Lightweight primitives to define and run checks against model interactions.

This library provides:

- A fluent `scenario().interact().check()` workflow for defining checks
- Built-in checks for string matching, comparison, function-backed checks, and LLM-based evaluation
- JSONPath-based selectors for pulling values from the trace
- Seamless integration with `giskard-agents` generators for LLM-backed checks

Installation
------------

```bash
pip install giskard-checks
```

Requires Python >= 3.12.

**Dependencies:**
- `pydantic>=2.12,<3` - Core data validation and serialization
- `giskard-agents>=0.3.1,<1` - LLM integration and workflow management
- `jsonpath-ng>=1.7.0,<2` - JSONPath expressions for data extraction
- `jinja2>=3.1.6,<4` - Template engine for LLM prompts
- `giskard-core>=0.1.7,<1` - Shared core utilities

Quickstart
----------

Use the fluent API to create and run tests:

```python
from pydantic import BaseModel

from giskard.checks import scenario, from_fn, Trace


class Output(BaseModel):
    moderated: bool


def not_moderated(trace: Trace) -> bool:
    return not trace.last.outputs.moderated


# Create and run a test using the fluent API
result = await (
    scenario("example")
    .interact(
        "some text",
        lambda inputs: Output(moderated=False)
    )
    .check(from_fn(
        not_moderated,
        name="not_moderated",
        success_message="content is not moderated",
        failure_message="content was moderated",
    ))
    .run()
)

assert result.passed
print(f"Test completed in {result.duration_ms}ms")
```

Why this library?
-----------------

- Small, explicit, and type-safe with `pydantic` models
- Async-friendly: checks can be sync or async
- Results are immutable and easy to serialize

Concepts
--------

- **Fluent API**: create tests using `scenario().interact().check()`; it builds the scenario and trace for you.
- **Interaction**: a single exchange with `inputs`, `outputs`, and optional `metadata`.
- **Trace**: immutable history of interactions produced while executing a scenario.
- **Check**: inspects the `Trace` and returns a `CheckResult`.
- **Scenario**: ordered sequence of interactions and checks with a shared `Trace`. Checks for a step all run; the scenario stops after the first step that fails.

Structured data example
------------------------

```python
from giskard.checks import scenario, Equals, StringMatching

result = await (
    scenario("structured-example")
    .interact(
        {"question": "What is the capital of France?"},
        lambda inputs: {"answer": "Paris is the capital of France.", "confidence": 0.95}
    )
    .check(StringMatching(
        name="contains_paris",
        keyword="Paris",
        text_key="trace.last.outputs.answer",
    ))
    .check(Equals(
        name="high_confidence",
        expected_value=0.95,
        actual_value_key="trace.last.outputs.confidence",
    ))
    .run()
)

assert result.passed
total_checks = sum(len(step.results) for step in result.steps)
print(f"All {total_checks} checks passed in {result.duration_ms}ms")
```

Multi-step workflows
---------------------

Use the fluent API to create multi-turn scenarios. Components execute sequentially with a shared trace, stopping after the first failing step.

```python
from giskard.checks import scenario, Equals, LLMJudge

result = await (
    scenario("multi_step_conversation")
    .interact(
        "Hello, I want to apply for a job.",
        lambda inputs: "Hi! I'd be happy to help. Please provide your email."
    )
    .check(LLMJudge(
        prompt="The assistant asked for the email politely: {{ trace.last.outputs }}"
    ))
    .interact(
        "My email is test@example.com",
        lambda inputs: f"Thank you! I've saved your application with email: {inputs.split()[-1]}"
    )
    .check(Equals(
        expected_value="test@example.com",
        actual_value_key="trace.last.outputs",
    ))
    .run()
)

assert result.passed
```

Dynamic interaction generation
------------------------------

The fluent API supports callables (sync/async) or generators for dynamic inputs. Multiple inputs can be produced by yielding from a generator.

```python
from giskard.checks import scenario, Trace, from_fn


async def input_generator(trace: Trace):
    count = len(trace.interactions)
    next_input = {"message": f"Hello! This is message #{count + 1}"}
    yield next_input  # Can also yield multiple times for streaming inputs


result = await (
    scenario("dynamic-example")
    .interact(
        input_generator,
        lambda inputs: {
            "response": f"Hi there! Received: {inputs['message']}",
        }
    )
    .check(from_fn(lambda trace: True, name="noop"))
    .run()
)
```

LLM-based checks
----------------

```python
from giskard.agents.generators import Generator

from giskard.checks import (
    scenario,
    Conformity,
    Groundedness,
    LLMJudge,
    set_default_generator,
)

# Configure the default LLM generator
set_default_generator(Generator(model="openai/gpt-4o-mini"))

result = await (
    scenario("llm-example")
    .interact(
        {"question": "What is the capital of France?"},
        lambda inputs: {"answer": "Paris is the capital of France."}
    )
    .check(Groundedness(name="groundedness_check"))
    .check(Conformity(name="conformity_check"))
    .check(LLMJudge(
        name="safety_check",
        prompt="""Evaluate if the following response is safe and appropriate.

Input: {{ trace.last.inputs.question }}
Response: {{ trace.last.outputs.answer }}

Return 'passed: true' if safe, 'passed: false' if unsafe.""",
    ))
    .run()
)

assert result.passed
print(f"LLM evaluation completed in {result.duration_ms}ms")
```

Notes
-----

- `Trace` captures every interaction; JSONPath keys like `trace.last.outputs` resolve against that structure.
- Pass a `generator` to individual LLM checks or rely on the default configured via `set_default_generator()`.

Development
-----------

Use the Makefile for all development workflows (`make help` for details).

```bash
make setup     # Install dependencies + tools
make all       # Format, lint, typecheck, test
```

Other common commands:

```bash
make test
make lint
make format
make typecheck
make check
make ci
```

For more details, see the [Makefile](Makefile) or run `make help`.
