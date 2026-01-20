=========
Scenarios
=========

Multi-step workflow testing with scenarios and test cases.

.. currentmodule:: giskard.checks


Scenario
--------

.. autoclass:: Scenario
   :members:
   :undoc-members:
   :show-inheritance:

Execute a sequence of interaction specs and checks with a shared trace.

**Attributes:**

* ``name``: Scenario name
* ``sequence``: List of components (InteractionSpecs and Checks)

**Methods:**

* ``run()``: Execute the scenario and return results

**Example:**

.. code-block:: python

   from giskard.checks import Scenario, InteractionSpec, from_fn

   scenario = Scenario(
       name="conversation_flow",
       sequence=[
           InteractionSpec(inputs="Hello", outputs="Hi!"),
           from_fn(lambda trace: "Hi" in trace.interactions[-1].outputs, name="check1"),
           InteractionSpec(inputs="How are you?", outputs="I'm well!"),
           from_fn(lambda trace: len(trace.interactions) == 2, name="check2"),
       ]
   )

   result = await scenario.run()
   print(f"Passed: {result.passed}")


ScenarioResult
--------------

.. autoclass:: ScenarioResult
   :members:
   :undoc-members:
   :show-inheritance:

Result of scenario execution.

**Attributes:**

* ``passed``: Whether all checks passed
* ``trace``: Final trace with all interactions
* ``results``: List of CheckResult objects
* ``duration_ms``: Total execution time in milliseconds
* ``error``: Optional error message if execution failed


TestCase
--------

.. autoclass:: TestCase
   :members:
   :undoc-members:
   :show-inheritance:

High-level API for running one interaction with multiple checks.

**Attributes:**

* ``name``: Test case name
* ``interaction``: InteractionSpec to execute
* ``checks``: List of checks to run

**Methods:**

* ``run()``: Execute the test case and return results

**Example:**

.. code-block:: python

   from giskard.checks import TestCase, InteractionSpec, StringMatchingCheck

   tc = TestCase(
       name="qa_test",
       interaction=InteractionSpec(
           inputs="What is the capital of France?",
           outputs=lambda inputs: "Paris"
       ),
       checks=[
           StringMatchingCheck(
               name="contains_paris",
               content="Paris",
               key="interactions[-1].outputs"
           )
       ]
   )

   result = await tc.run()
   assert result.passed


TestCaseResult
--------------

.. autoclass:: TestCaseResult
   :members:
   :undoc-members:
   :show-inheritance:

Result of test case execution.

**Attributes:**

* ``passed``: Whether all checks passed
* ``results``: List of CheckResult objects
* ``duration_ms``: Total execution time
* ``error``: Optional error message


Runners
-------

ScenarioRunner
~~~~~~~~~~~~~~

.. autoclass:: ScenarioRunner
   :members:
   :undoc-members:
   :show-inheritance:

Low-level runner for executing scenarios.

**Example:**

.. code-block:: python

   from giskard.checks import ScenarioRunner, Scenario

   runner = ScenarioRunner()
   result = await runner.run(scenario)


TestCaseRunner
~~~~~~~~~~~~~~

.. autoclass:: TestCaseRunner
   :members:
   :undoc-members:
   :show-inheritance:

Low-level runner for executing test cases.


Usage Patterns
--------------

Running Multiple Test Cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from giskard.checks import TestCase

   test_cases = [
       TestCase(name="test1", interaction=..., checks=[...]),
       TestCase(name="test2", interaction=..., checks=[...]),
   ]

   async def run_all():
       results = []
       for tc in test_cases:
           result = await tc.run()
           results.append((tc.name, result))

       passed = sum(1 for _, r in results if r.passed)
       print(f"Passed: {passed}/{len(results)}")

   asyncio.run(run_all())


Scenario with Dynamic Interactions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from giskard.checks import Scenario, InteractionSpec, Trace

   async def dynamic_input(trace: Trace):
       count = len(trace.interactions)
       return f"Message #{count + 1}"

   scenario = Scenario(
       name="dynamic_flow",
       sequence=[
           InteractionSpec(
               inputs=dynamic_input,
               outputs=lambda inputs: f"Echo: {inputs}"
           ),
           InteractionSpec(
               inputs=dynamic_input,
               outputs=lambda inputs: f"Echo: {inputs}"
           ),
       ]
   )


Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   try:
       result = await tc.run()
       if not result.passed:
           print(f"Test failed: {result.error or 'See individual check results'}")
           for check_result in result.results:
               if not check_result.passed:
                   print(f"  - {check_result.name}: {check_result.message}")
   except Exception as e:
       print(f"Test execution error: {e}")
