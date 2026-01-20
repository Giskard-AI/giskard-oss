====
Core
====

Core types and base classes for building tests and checks.

.. currentmodule:: giskard.checks


Check
-----

.. autoclass:: Check
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Base class for all checks. Subclass and register with ``@Check.register("kind")`` to create custom checks.

**Example:**

.. code-block:: python

   from giskard.checks import Check, CheckResult, Trace

   @Check.register("my_check")
   class MyCheck(Check):
       threshold: float = 0.8

       async def run(self, trace: Trace) -> CheckResult:
           # Your check logic
           return CheckResult.success("Check passed")


CheckResult
-----------

.. autoclass:: CheckResult
   :members:
   :undoc-members:
   :show-inheritance:

Result of a check execution with status, message, and optional metrics.

**Example:**

.. code-block:: python

   # Success
   result = CheckResult.success(
       message="Check passed",
       metrics={"score": 0.95}
   )

   # Failure
   result = CheckResult.failure(
       message="Check failed",
       details={"reason": "threshold not met"}
   )


CheckStatus
-----------

.. autoclass:: CheckStatus
   :members:
   :undoc-members:

Enumeration of possible check statuses: ``PASSED``, ``FAILED``, ``ERROR``.


Interaction
-----------

.. autoclass:: Interaction
   :members:
   :undoc-members:
   :show-inheritance:

A single exchange between inputs and outputs.

**Example:**

.. code-block:: python

   from giskard.checks import Interaction

   interaction = Interaction(
       inputs="What is 2+2?",
       outputs="4",
       metadata={"model": "gpt-4", "tokens": 5}
   )


Trace
-----

.. autoclass:: Trace
   :members:
   :undoc-members:
   :show-inheritance:

Immutable history of all interactions in a scenario.

**Example:**

.. code-block:: python

   from giskard.checks import Trace, Interaction

   trace = Trace(interactions=[
       Interaction(inputs="Hello", outputs="Hi!"),
       Interaction(inputs="How are you?", outputs="I'm well!"),
   ])

   # Access last interaction
   last = trace.interactions[-1]


InteractionSpec
---------------

.. autoclass:: InteractionSpec
   :members:
   :undoc-members:
   :show-inheritance:

Declarative specification for generating interactions.

**Example:**

.. code-block:: python

   from giskard.checks import InteractionSpec

   # Static values
   spec = InteractionSpec(
       inputs="test input",
       outputs="test output"
   )

   # Callable outputs
   spec = InteractionSpec(
       inputs="test",
       outputs=lambda inputs: my_function(inputs)
   )


BaseInteractionSpec
-------------------

.. autoclass:: BaseInteractionSpec
   :members:
   :undoc-members:
   :show-inheritance:

Base class for custom interaction specifications.


Scenario
--------

.. autoclass:: Scenario
   :members:
   :undoc-members:
   :show-inheritance:

Ordered sequence of interaction specs and checks with shared trace.

**Example:**

.. code-block:: python

   from giskard.checks import Scenario, InteractionSpec, from_fn

   scenario = Scenario(
       name="test_flow",
       sequence=[
           InteractionSpec(inputs="hello", outputs="hi"),
           from_fn(lambda trace: True, name="check1"),
           InteractionSpec(inputs="bye", outputs="goodbye"),
       ]
   )

   result = await scenario.run()


TestCase
--------

.. autoclass:: TestCase
   :members:
   :undoc-members:
   :show-inheritance:

Convenience wrapper for running one interaction with multiple checks.

**Example:**

.. code-block:: python

   from giskard.checks import TestCase, InteractionSpec, from_fn

   tc = TestCase(
       name="my_test",
       interaction=InteractionSpec(inputs="test", outputs="result"),
       checks=[
           from_fn(lambda trace: True, name="check1"),
           from_fn(lambda trace: True, name="check2"),
       ]
   )

   result = await tc.run()


Extractors
----------

.. autoclass:: Extractor
   :members:
   :undoc-members:
   :show-inheritance:

Base class for extracting values from traces.


.. autoclass:: JsonPathExtractor
   :members:
   :undoc-members:
   :show-inheritance:

Extract values using JSONPath expressions.

**Example:**

.. code-block:: python

   from giskard.checks import JsonPathExtractor

   extractor = JsonPathExtractor(key="interactions[-1].outputs.answer")
   value = extractor.extract(trace)


Configuration
-------------

.. autofunction:: set_default_generator

Set the default LLM generator for LLM-based checks.

**Example:**

.. code-block:: python

   from giskard.agents.generators import Generator
   from giskard.checks import set_default_generator

   set_default_generator(Generator(model="openai/gpt-4o-mini"))


.. autofunction:: get_default_generator

Get the currently configured default generator.


ScenarioRunner
--------------

.. autoclass:: ScenarioRunner
   :members:
   :undoc-members:
   :show-inheritance:

Low-level runner for executing scenarios.
