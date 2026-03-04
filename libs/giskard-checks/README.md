giskard-checks
===============

Lightweight primitives to define and run checks against model interactions.

This library provides:

- Core types for describing interactions (`Interact`, `Interaction`, `Trace`)
- A fluent scenario builder and runner (`scenario`, `Scenario`, `ScenarioResult`)
- Built-in checks including string matching, comparisons, and LLM-based evaluation
- JSONPath-based extraction utilities for referencing trace data
- Seamless integration with `giskard-agents` generators for LLM-backed checks

Installation
------------

```bash
pip install giskard-checks
```

Requires Python >= 3.12.

**Dependencies:**
- `pydantic>=2.11.7` - Core data validation and serialization
- `giskard-agents>=0.3` - LLM integration and workflow management
- `jsonpath-ng>=1.7.0` - JSONPath expressions for data extraction
- `jinja2>=3.1.6` - Template engine for LLM prompts

Quickstart
----------

Use the fluent API to create and run scenarios:

```python
from giskard.checks import Groundedness, scenario

test_scenario = (
    scenario("test_france_capital")
    .interact(
        inputs="What is the capital of France?",
        outputs="The capital of France is Paris."
    )
    .check(
        Groundedness(
            name="answer is grounded",
            answer_key="trace.last.outputs",
            context="""France is a country in Western Europe. Its capital
                       and largest city is Paris, known for the Eiffel Tower
                       and the Louvre Museum."""
        )
    )
)

result = await test_scenario.run()
assert result.passed
print(f"Scenario completed in {result.duration_ms}ms")
```

The fluent API accepts static values or callables for `inputs` and `outputs`, so you can call your SUT directly:

```python
from openai import OpenAI
from giskard.checks import Groundedness, scenario

client = OpenAI()

def get_answer(inputs: str) -> str:
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": inputs}],
    )
    return response.choices[0].message.content

test_scenario = (
    scenario("test_dynamic_output")
    .interact(
        inputs="What is the capital of France?",
        outputs=get_answer
    )
    .check(
        Groundedness(
            name="answer is grounded",
            answer_key="trace.last.outputs",
            context="France is a country in Western Europe..."
        )
    )
)
```

The `run()` method is async. In a script, wrap it with `asyncio.run()`:

```python
import asyncio

async def main():
    result = await test_scenario.run()
    print(result)

asyncio.run(main())
```

Why this library?
-----------------

- Small, explicit, and type-safe with `pydantic` models
- Async-friendly: checks can be sync or async
- Results are immutable and easy to serialize

Concepts
--------

- **Fluent API**: The recommended way to create tests using `scenario().interact().check()`. This API builds a scenario and handles interaction generation.
- **Interact**: A specification for generating interactions dynamically (static values, callables, or generators).
- **Trace**: Immutable history of all `Interaction` objects produced while executing a scenario. Use `trace.last` in JSONPath expressions (e.g., `trace.last.outputs`).
- **Interaction**: A recorded exchange with `inputs`, `outputs`, and optional `metadata`.
- **Check**: Inspects the `Trace` and returns a `CheckResult`.
- **Scenario**: Ordered sequence of interactions and checks with a shared `Trace`. Execution stops at the first failing check and later steps are skipped.

**Advanced concepts** (used internally by the fluent API):
- **TestCase**: Wrapper that runs a set of checks against a single trace step and returns a `TestCaseResult`.
- **ScenarioRunner**: Executes scenarios sequentially, maintaining trace state and aggregating step results.

API Overview
------------

**Core types**
- `giskard.checks.Check`: base class for all checks with discriminated-union registration.
- `giskard.checks.CheckResult`, `CheckStatus`, `Metric`: typed results with convenience helpers.
- `giskard.checks.Trace` / `Interaction`: a trace is an immutable sequence of recorded interactions with the system.
- `giskard.checks.Scenario` and `ScenarioResult`: ordered sequence of components with shared trace. Execution stops at first failure and later steps are skipped.
- `giskard.checks.TestCase` and `TestCaseResult`: runs checks against a trace step and aggregates results.

**Interaction specs**
- `giskard.checks.InteractionSpec`: discriminated base for describing inputs/outputs. Subclasses implement `generate()` to yield interactions.
- `giskard.checks.Interact`: batteries-included spec that supports static values, callables, or generators for both inputs and outputs. Supports multi-turn interactions via generators.

**Scenarios and runners**
- `giskard.checks.Scenario`: ordered sequence of components (InteractionSpecs and Checks) with shared trace. Components execute sequentially, stopping at first failure.
- `giskard.checks.ScenarioRunner`: executes scenarios with timing, error capture, and early-stop semantics.
- `giskard.checks.TestCaseRunner`: executes test cases with timing and error handling.

**Built-in and LLM-based checks**
- `giskard.checks.from_fn`, `FnCheck`: wrap arbitrary callables.
- `giskard.checks.StringMatching`, `RegexMatching`, `SemanticSimilarity`, `Equals`, `NotEquals`, `GreaterThan`, `GreaterEquals`, `LesserThan`, `LesserThanEquals`.
- `giskard.checks.BaseLLMCheck`, `LLMCheckResult`, `Groundedness`, `Conformity`, `LLMJudge`.
- JSONPath selectors (e.g., `trace.last.outputs`) are supported on relevant checks via `key` or check-specific fields like `answer_key`.

**Testing utilities**
- `giskard.checks.WithSpy`: wrapper for spying on function calls during interaction generation.

**Settings**
- `giskard.checks.set_default_generator` / `get_default_generator`: configure the generator used by LLM checks.

Testing
-------

- Tests live under `tests/` mirroring the package structure (`tests/core`, `tests/scenarios`, `tests/trace`).
- Use `make test` (or `make ci`) to run the full suite exactly as CI does.

Usage Notes
-----------

- Define custom checks with a unique `KIND` via `@Check.register("kind")`.
- All discriminated types auto-register when imported; ensure modules are imported before deserialization.
- Prefer `model_dump()` / `model_validate()` for serialization.
- Attach extra metadata in `CheckResult.details`; JSONPath helpers (`key=...`) resolve against the entire trace.

Serialization
-------------

The library uses Pydantic's discriminated unions for polymorphic serialization.

```python
from giskard.checks import Check, CheckResult, Interaction, TestCase, Trace


@Check.register("my_custom_check")
class MyCustomCheck(Check):
    async def run(self, trace: Trace) -> CheckResult:
        return CheckResult.success("Check passed")


trace = Trace(interactions=[Interaction(inputs="test", outputs="result")])
check = MyCustomCheck(name="test")
testcase = TestCase(trace=trace, checks=[check], name="example")

# Serialize to dict
serialized = testcase.model_dump()

# Deserialize back (requires classes to be imported)
restored = TestCase.model_validate(serialized)
```

**Important**: Import every custom type (checks and specs) before calling `model_validate()`. The registry only knows about classes already loaded into memory.

Creating Custom Checks and Interaction Specs
--------------------------------------------

### Step 1: Define a custom check

```python
from giskard.checks import Check, CheckResult, Trace


@Check.register("advanced_security")
class AdvancedSecurityCheck(Check):
    threshold: float = 0.8

    async def run(self, trace: Trace) -> CheckResult:
        current = trace.last
        score = await some_security_analysis(current.outputs)
        if score >= self.threshold:
            return CheckResult.success(f"Security score {score:.2f} meets threshold")
        return CheckResult.failure(
            f"Security score {score:.2f} below threshold {self.threshold}"
        )
```

### Step 2: Define a custom interaction specification

```python
from giskard.checks import InteractionSpec, Interaction, Trace


@InteractionSpec.register("chat_conversation")
class ChatInteraction(InteractionSpec):
    session_id: str
    messages: list[str]

    async def generate(self, trace: Trace):
        summary = f"Conversation with {len(self.messages)} messages"
        record = Interaction(
            inputs=self.messages,
            outputs={"summary": summary},
            metadata={"session_id": self.session_id},
        )
        yield record
```

### Step 3: Verify registration

```python
from giskard.checks import Scenario

chat = ChatInteraction(session_id="session_123", messages=["hi", "hello"])
check = AdvancedSecurityCheck(name="security_test", threshold=0.7)
scenario = Scenario(name="custom_test", sequence=[chat, check])

serialized = scenario.model_dump()
restored = Scenario.model_validate(serialized)
```

Troubleshooting Serialization Issues
------------------------------------

**ValidationError**: "Kind is not provided for Check"
- Cause: Custom class not imported before deserialization.
- Fix: Import classes before calling `model_validate()`.

**DuplicateKindError**: "Duplicate kind 'my_check' detected"
- Cause: Two classes share the same `KIND`.
- Fix: Give every registered class a unique `KIND`.

**Missing registration**
- Cause: Subclass missing the decorator.
- Fix: Use `@Check.register("...")` (or the relevant base).

**Import order issues in tests**
- Cause: Tests call `model_validate()` before importing custom modules.
- Fix: Import those modules in test setup or fixtures first.

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
        key="trace.last.outputs.confidence",
    ))
    .run()
)

assert result.passed
print(f"Scenario completed in {result.duration_ms}ms")
```

Multi-step workflows
---------------------

Use the fluent API to create multi-turn scenarios. Components execute sequentially with a shared trace, stopping at the first failing check.

```python
from giskard.checks import scenario, LLMJudge, RegexMatching

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
    .check(RegexMatching(
        pattern="test@example.com",
        text_key="trace.last.outputs",
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
    .check(Conformity(name="conformity_check", rule="The agent must answer in English"))
    .check(LLMJudge(
        name="safety_check",
        prompt="""Evaluate if the following response is safe and appropriate.

Input: {{ trace.last.inputs }}
Response: {{ trace.last.outputs }}

Return 'passed: true' if safe, 'passed: false' if unsafe.""",
    ))
    .run()
)

assert result.passed
print(f"LLM evaluation completed in {result.duration_ms}ms")
```

Template customization & advanced LLM usage
-------------------------------------------

- LLM-based checks ship with template references registered inside `giskard.agents`.
- Provide your own template by overriding `get_prompt()` in a subclass or by instantiating `LLMJudge` with inline prompts.
- Templates use the same interpolation context you return from `get_inputs()`.

```python
from giskard.agents.workflow import TemplateReference
from pydantic import BaseModel

from giskard.checks import BaseLLMCheck, Check, CheckResult, Trace


class CustomResult(BaseModel):
    score: float
    passed: bool
    reasoning: str


@Check.register("custom_llm_check")
class CustomLLMCheck(BaseLLMCheck):
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(template_name="my_project::checks/custom_check.j2")

    @property
    def output_type(self) -> type[BaseModel]:
        return CustomResult

    async def _handle_output(
        self,
        output_value: CustomResult,
        template_inputs: dict[str, str],
        trace: Trace,
    ) -> CheckResult:
        if output_value.score >= 0.8:
            return CheckResult.success(f"Score {output_value.score} meets threshold")
        return CheckResult.failure(f"Score {output_value.score} below threshold")
```

Notes
-----

- `Trace` captures every interaction; JSONPath keys like `trace.last.outputs` resolve against that structure.
- Pass a `generator` to individual LLM checks or rely on the default configured via `set_default_generator()`.
- Built-in LLM checks rely on templates bundled in `giskard.checks` and registered with the `giskard-agents` template system; override `get_prompt` or `get_inputs` for customization.

Advanced Usage
--------------

For advanced use cases where you need direct control over interactions or trace construction, you can build a `Trace` for `TestCase` directly, using `Interaction`:

``` python
from giskard.checks import Interaction, TestCase, Trace

# Build a Trace manually for a TestCase
trace = Trace(interactions=[
    Interaction(inputs="some text", outputs=process("some text")),
])
tc = TestCase(trace=trace, checks=[check1, check2], name="advanced_example")
test_case_result = await tc.run()
```

For programmatic test generation or when you need fine-grained control, you can also construct `Scenario` objects directly, creating a sequence of `InteractionSpec` or `Check` objects:

```python
from giskard.checks import (
    Scenario,
    Interact, # Inherits from `InteractionSpec`
    Equals # Inherits from `Check`
)

scenario = Scenario(
    name="programmatic_scenario",
    sequence=[
        Interact(inputs="Hello", outputs=lambda inputs: "Hi"),
        Equals(expected="Hi", key="trace.last.outputs"),
    ]
)

result = await scenario.run()
```

**Note**: For most use cases, the fluent API (`scenario().interact().check()`) is recommended as it's simpler and more readable.

Development
-----------

Use the Makefile for all development workflows (`make help` for details).

```bash
make install   # Install dependencies
make setup     # Install dependencies + tools (Format, lint, typecheck, test)
```

Other common commands:

```bash
make test
make lint
make format
make typecheck
make check
make clean
```

For more details, see the [Makefile](Makefile) or run `make help`.
