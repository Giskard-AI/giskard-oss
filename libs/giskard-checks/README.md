giskard-checks
===============

Lightweight primitives to define and run checks against model interactions.

This library provides:

- Core types for describing interactions (`InteractionSpec`, `Trace`, `ScenarioComponent`)
- A scenario runner with aggregated results (`ScenarioRunner`, `TestCaseResult`)
- Built-in checks including string matching, equality, extraction, and LLM-based evaluation
- JSONPath-based extraction utilities and composable extractor interfaces
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

Use the fluent API to create and run tests:

```python
from pydantic import BaseModel

from giskard.checks import scenario, from_fn, Trace


class Output(BaseModel):
    moderated: bool


def not_moderated(trace: Trace) -> bool:
    return not trace.interactions[-1].outputs.moderated


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

- **Fluent API**: The recommended way to create tests using `scenario().interact().check()`. This API automatically handles interaction generation and scenario construction.
- **Interaction**: a single exchange with `inputs`, `outputs`, and optional `metadata`.
- **Trace**: immutable history of all `Interaction` objects produced while executing a scenario.
- **Check**: inspects the `Trace` and returns a `CheckResult`.
- **Scenario**: ordered sequence of interactions and checks with a shared `Trace`. Execution stops at the first failing check.

**Advanced concepts** (used internally by the fluent API):
- **InteractionSpec**: declarative description of how to produce interactions (static values, callables, or generators).
- **TestCase**: convenience wrapper that runs `[interaction, checks...]` using the runner and surfaces `TestCaseResult`.
- **ScenarioComponent**: polymorphic base for both specs and checks; components are executed in order.
- **ScenarioRunner**: executes scenarios by processing components sequentially, maintaining trace state.

API Overview
------------

**Core types**
- `giskard.checks.Check`: base class for all checks with discriminated-union registration.
- `giskard.checks.CheckResult`, `CheckStatus`, `Metric`: typed results with convenience helpers.
- `giskard.checks.Interaction` / `Trace`: immutable interaction payloads plus accumulated history.
- `giskard.checks.Scenario` and `ScenarioResult`: ordered sequence of components (InteractionSpecs and Checks) with shared trace. Execution stops at first failure.
- `giskard.checks.TestCase` and `TestCaseResult`: high-level API for `[InteractionSpec + checks]` with aggregate of multi-run executions.

**Interaction specs**
- `giskard.checks.BaseInteractionSpec`: discriminated base for describing inputs/outputs. Subclasses implement `generate()` to yield interactions.
- `giskard.checks.InteractionSpec`: batteries-included spec that supports static values, callables, or generators for both inputs and outputs. Supports multi-turn interactions via generators.

**Scenarios and runners**
- `giskard.checks.Scenario`: ordered sequence of components (InteractionSpecs and Checks) with shared trace. Components execute sequentially, stopping at first failure.
- `giskard.checks.ScenarioRunner`: executes both scenarios and test cases with timing, error capture, and early-stop semantics.
- `giskard.checks.TestCaseRunner`: executes test cases with timing and error handling.

**Built-in checks**
- `giskard.checks.from_fn`, `FnCheck`: wrap arbitrary callables.
- `giskard.checks.StringMatching`, `Equality`, `ExtractionCheck`.
- `giskard.checks.BaseLLMCheck`, `LLMCheckResult`, `Groundedness`, `Conformity`, `LLMJudge`.
- All extraction-capable checks share JSONPath selectors via `key` or custom `Extractor`s.

**Extraction utilities**
- `giskard.checks.Extractor`, `JsonPathExtractor`: base classes for extracting values from traces.

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
- Environment variable `GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS` controls duplicate-kind enforcement (enabled by default).

Serialization
-------------

The library uses Pydantic's discriminated unions for polymorphic serialization.

```python
from giskard.checks import Check, CheckResult, InteractionSpec, TestCase, Trace


@Check.register("my_custom_check")
class MyCustomCheck(Check):
    async def run(self, trace: Trace) -> CheckResult:
        return CheckResult.success("Check passed")


interaction = InteractionSpec(inputs="test", outputs="result")
check = MyCustomCheck(name="test")
testcase = TestCase(interaction=interaction, checks=[check], name="example")

# Serialize to dict
serialized = testcase.model_dump()

# Deserialize back (requires classes to be imported)
restored = TestCase.model_validate(serialized)
```

**Important**: Import every custom type (checks, specs, extractors) before calling `model_validate()`. The registry only knows about classes already loaded into memory.

Creating Custom Checks and Interaction Specs
--------------------------------------------

### Step 1: Define a custom check

```python
from giskard.checks import Check, CheckResult, Trace


@Check.register("advanced_security")
class AdvancedSecurityCheck(Check):
    threshold: float = 0.8

    async def run(self, trace: Trace) -> CheckResult:
        current = trace.interactions[-1]
        score = await some_security_analysis(current.outputs)
        if score >= self.threshold:
            return CheckResult.success(f"Security score {score:.2f} meets threshold")
        return CheckResult.failure(
            f"Security score {score:.2f} below threshold {self.threshold}"
        )
```

### Step 2: Define a custom interaction specification

```python
from giskard.checks import BaseInteractionSpec, Interaction, Trace


@BaseInteractionSpec.register("chat_conversation")
class ChatInteraction(BaseInteractionSpec):
    session_id: str
    messages: list[str]

    async def handle(self, trace: Trace):
        summary = f"Conversation with {len(self.messages)} messages"
        interaction = Interaction(
            inputs=self.messages,
            outputs={"summary": summary},
            metadata={"session_id": self.session_id},
        )
        yield interaction
```

### Step 3: Verify registration

```python
from giskard.checks import TestCase

chat = ChatInteraction(session_id="session_123", messages=["hi", "hello"])
check = AdvancedSecurityCheck(name="security_test", threshold=0.7)
testcase = TestCase(interaction=chat, checks=[check], name="custom_test")

serialized = testcase.model_dump()
restored = TestCase.model_validate(serialized)
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

Environment variables:

- `GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS=1` (default): raises on duplicates.
- `GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS=0`: logs a warning and last definition wins.

Structured data example
------------------------

```python
from giskard.checks import scenario, Equality, StringMatching

result = await (
    scenario("structured-example")
    .interact(
        {"question": "What is the capital of France?"},
        lambda inputs: {"answer": "Paris is the capital of France.", "confidence": 0.95}
    )
    .check(StringMatching(
        name="contains_paris",
        content="Paris",
        key="interactions[-1].outputs.answer",
    ))
    .check(Equality(
        name="high_confidence",
        expected=0.95,
        key="interactions[-1].outputs.confidence",
    ))
    .run()
)

assert result.passed
print(f"All {len(result.results)} checks passed in {result.duration_ms}ms")
```

Multi-step workflows
---------------------

Use the fluent API to create multi-turn scenarios. Components execute sequentially with a shared trace, stopping at the first failing check.

```python
from giskard.checks import scenario, Equality, LLMJudge

result = await (
    scenario("multi_step_conversation")
    .interact(
        "Hello, I want to apply for a job.",
        lambda inputs: "Hi! I'd be happy to help. Please provide your email."
    )
    .check(LLMJudge(
        prompt="The assistant asked for the email politely: {{ interactions[-1].outputs }}"
    ))
    .interact(
        "My email is test@example.com",
        lambda inputs: f"Thank you! I've saved your application with email: {inputs.split()[-1]}"
    )
    .check(Equality(
        expected="test@example.com",
        key="interactions[-1].outputs",
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

Input: {{ inputs.question }}
Response: {{ outputs.answer }}

Return 'passed: true' if safe, 'passed: false' if unsafe.""",
    ))
    .run()
)

assert result.passed
print(f"LLM evaluation completed in {result.duration_ms}ms")
```

Template customization & advanced LLM usage
-------------------------------------------

- Built-in checks ship with template references registered inside `giskard.agents`.
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

- `Trace` captures every interaction; JSONPath keys like `interactions[-1].outputs` resolve against that structure.
- `StringMatching` supports `evaluation_mode="any" | "all" | "none"` for lists.
- Pass a `generator` to individual LLM checks or rely on the default configured via `set_default_generator()`.
- Built-in LLM checks rely on templates bundled with `giskard-agents`; override `get_prompt` or `get_inputs` for customization.

Advanced Usage
--------------

For advanced use cases where you need direct control over interaction specs or test cases, you can use `InteractionSpec` and `TestCase` directly:

```python
from giskard.checks import InteractionSpec, TestCase, from_fn, Trace

# Create an InteractionSpec manually
interaction = InteractionSpec(
    inputs="some text",
    outputs=lambda inputs: process(inputs),
)

# Create a TestCase manually
tc = TestCase(
    interaction=interaction,
    checks=[check1, check2],
    name="advanced_example"
)

result = await tc.run()
```

For programmatic test generation or when you need fine-grained control, you can also construct `Scenario` objects directly:

```python
from giskard.checks import Scenario, InteractionSpec, Equality

scenario = Scenario(
    name="programmatic_scenario",
    sequence=[
        InteractionSpec(inputs="Hello", outputs=lambda inputs: "Hi"),
        Equality(expected="Hi", key="interactions[-1].outputs"),
    ]
)

result = await scenario.run()
```

**Note**: For most use cases, the fluent API (`scenario().interact().check()`) is recommended as it's simpler and more readable.

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
