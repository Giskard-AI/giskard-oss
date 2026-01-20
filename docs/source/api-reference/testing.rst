================
Testing Utilities
================

Utilities for testing and debugging.

.. currentmodule:: giskard.checks


WithSpy
-------

.. autoclass:: WithSpy
   :members:
   :undoc-members:
   :show-inheritance:

Wrapper for spying on function calls during interaction generation.

**Example:**

.. code-block:: python

   from giskard.checks import WithSpy

   def my_function(x: int) -> int:
       return x * 2

   spy = WithSpy(my_function)
   result = spy(5)

   print(f"Called with: {spy.calls}")
   print(f"Result: {result}")


Usage in Tests
--------------

Spying on Model Calls
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from giskard.checks import InteractionSpec, TestCase, WithSpy, from_fn

   # Wrap your model
   model_spy = WithSpy(my_llm_model)

   interaction = InteractionSpec(
       inputs="test input",
       outputs=lambda inputs: model_spy(inputs)
   )

   check = from_fn(
       lambda trace: len(model_spy.calls) == 1,
       name="single_call",
       success_message="Model called exactly once"
   )

   tc = TestCase(interaction=interaction, checks=[check], name="spy_test")
   result = await tc.run()


Debugging Helpers
-----------------

Inspecting Traces
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from giskard.checks import Trace

   def print_trace(trace: Trace):
       """Helper to inspect trace contents."""
       print(f"Trace with {len(trace.interactions)} interactions:")
       for i, interaction in enumerate(trace.interactions, 1):
           print(f"\n{i}. Interaction:")
           print(f"   Inputs: {interaction.inputs}")
           print(f"   Outputs: {interaction.outputs}")
           print(f"   Metadata: {interaction.metadata}")


Debug Check
~~~~~~~~~~~

.. code-block:: python

   from giskard.checks import from_fn

   def debug_check(trace):
       """Check that prints trace for debugging."""
       print("\n=== Debug Trace ===")
       for i, interaction in enumerate(trace.interactions):
           print(f"Interaction {i}:")
           print(f"  Inputs: {interaction.inputs}")
           print(f"  Outputs: {interaction.outputs}")
       print("===================\n")
       return True

   check = from_fn(debug_check, name="debug")


Serialization Utilities
-----------------------

Saving and Loading Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import json
   from giskard.checks import TestCase, CheckResult

   # Run test
   tc = TestCase(...)
   result = await tc.run()

   # Serialize
   serialized = result.model_dump()
   with open("result.json", "w") as f:
       json.dump(serialized, f, indent=2)

   # Load
   with open("result.json", "r") as f:
       data = json.load(f)

   # Note: Can't directly validate back to TestCaseResult
   # but data is preserved


Custom Test Fixtures
--------------------

Reusable Test Setup
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import List
   from giskard.checks import TestCase, InteractionSpec, Check

   class TestFixture:
       def __init__(self, system_under_test):
           self.sut = system_under_test

       def create_test(
           self,
           name: str,
           input_text: str,
           checks: List[Check]
       ) -> TestCase:
           """Factory for creating test cases."""
           interaction = InteractionSpec(
               inputs=input_text,
               outputs=lambda inputs: self.sut(inputs)
           )
           return TestCase(
               name=name,
               interaction=interaction,
               checks=checks
           )

       async def run_test_suite(
           self,
           test_cases: List[TestCase]
       ):
           """Run multiple tests and report."""
           results = []
           for tc in test_cases:
               result = await tc.run()
               results.append((tc.name, result))
           return results


Parameterized Tests
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from giskard.checks import TestCase, InteractionSpec, StringMatchingCheck

   test_data = [
       ("What is 2+2?", "4"),
       ("What is 3+3?", "6"),
       ("What is 5+5?", "10"),
   ]

   @pytest.mark.parametrize("question,expected", test_data)
   @pytest.mark.asyncio
   async def test_calculator(question, expected):
       interaction = InteractionSpec(
           inputs=question,
           outputs=lambda inputs: calculator(inputs)
       )

       check = StringMatchingCheck(
           name="correct_answer",
           content=expected,
           key="interactions[-1].outputs"
       )

       tc = TestCase(
           interaction=interaction,
           checks=[check],
           name=f"calc_{expected}"
       )

       result = await tc.run()
       assert result.passed


Performance Tracking
--------------------

Measuring Execution Time
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from giskard.checks import TestCase

   async def benchmark_test(tc: TestCase, iterations: int = 10):
       """Run test multiple times and track performance."""
       durations = []

       for _ in range(iterations):
           result = await tc.run()
           durations.append(result.duration_ms)

       avg_duration = sum(durations) / len(durations)
       min_duration = min(durations)
       max_duration = max(durations)

       print(f"Average: {avg_duration:.2f}ms")
       print(f"Min: {min_duration:.2f}ms")
       print(f"Max: {max_duration:.2f}ms")


Tracking Metrics
~~~~~~~~~~~~~~~~

.. code-block:: python

   from giskard.checks import TestCase
   from typing import Dict, List

   class MetricsCollector:
       def __init__(self):
           self.metrics: List[Dict] = []

       async def run_and_collect(self, tc: TestCase):
           """Run test and collect metrics."""
           result = await tc.run()

           test_metrics = {
               "name": tc.name,
               "passed": result.passed,
               "duration_ms": result.duration_ms,
           }

           # Collect check-specific metrics
           for check_result in result.results:
               if check_result.metrics:
                   test_metrics.update({
                       f"{check_result.name}_{k}": v
                       for k, v in check_result.metrics.items()
                   })

           self.metrics.append(test_metrics)
           return result

       def summary(self):
           """Print summary of collected metrics."""
           if not self.metrics:
               return

           total = len(self.metrics)
           passed = sum(1 for m in self.metrics if m["passed"])
           avg_duration = sum(m["duration_ms"] for m in self.metrics) / total

           print(f"Tests: {passed}/{total} passed")
           print(f"Avg duration: {avg_duration:.2f}ms")
