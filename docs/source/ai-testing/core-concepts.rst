=============
Core Concepts
=============

Understanding the key concepts in Giskard Checks will help you write effective tests for your AI applications.


Overview
--------

Giskard Checks provides a fluent API that makes it easy to write tests. The recommended way to create tests is using ``scenario().interact().check()``:

.. code-block:: python

   from giskard.checks import scenario, StringMatchingCheck

   result = await (
       scenario("my_test")
       .interact("Hello", lambda inputs: my_bot(inputs))
       .check(StringMatchingCheck(content="Hi", key="interactions[-1].outputs"))
       .run()
   )
   print(f"Test passed: {result.passed}")

This fluent API handles all the underlying concepts automatically. For advanced use cases, you can work with the core primitives directly (see :ref:`advanced-concepts` below).


Core Concepts
=============

Interaction
-----------

An ``Interaction`` represents a single turn of data exchange with the system under test.

.. code-block:: python

   from giskard.checks import Interaction

   interaction = Interaction(
       inputs="What is the capital of France?",
       outputs="The capital of France is Paris.",
       metadata={"model": "gpt-4", "tokens": 15, "latency_ms": 234}
   )

**Properties:**

* ``inputs``: The input to your system (string, dict, Pydantic model, etc.)
* ``outputs``: The output from your system (any serializable type)
* ``metadata``: Optional dictionary for additional context (timings, model info, etc.)

Interactions are **immutable**, as they represent something that has already happened.


InteractionSpec
---------------

.. _advanced-concepts:

**Advanced Usage**: An ``InteractionSpec`` describes *how* to generate an interaction. Both inputs and outputs can be generated dynamically. In most cases, you'll use the fluent API (``scenario().interact()``) which creates ``InteractionSpec`` objects automatically.

For advanced use cases where you need direct control:

.. code-block:: python

   from giskard.checks import InteractionSpec
   from openai import OpenAI
   import random

   def generate_random_question() -> str:
       return f"What is 2 + {random.randint(0, 10)}?"

    def generate_answer(inputs: str) -> str:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": inputs}],
        )
        return response.choices[0].message.content

   spec = InteractionSpec(
       inputs=generate_random_question,
       outputs=generate_answer,
       metadata={
        "category": "math",
        "difficulty": "easy"
       }
   )

This is useful when you need fine-grained control over interaction generation, but for most use cases, the fluent API is simpler. See :doc:`multi-turn` for practical examples.

Trace
-----

A ``Trace`` is an immutable snapshot of all data exchanged with the system under test. In its simplest form, it is a list of interactions.

.. code-block:: python

   from giskard.checks import Trace, Interaction

   trace = Trace.from_interactions(
       Interaction(inputs="Hello", outputs="Hi there!"),
       Interaction(inputs="How are you?", outputs="I'm doing well, thanks!"),
   )

Traces can also be created from ``InteractionSpec`` objects. In that case, the generation is performed immediately to resolve each spec into a frozen interaction.


Checks
------

A ``Check`` validates something about a trace and returns a ``CheckResult``. There's a library of built-in checks , but you can also create your own.

.. code-block:: python

   from giskard.checks.builtin import Groundedness
   from giskard.checks import Trace

   check = Groundedness(
        answer_key="last.outputs",
        context="Giskard Checks is a testing framework for AI systems."
   )

TODO


Scenarios (Recommended)
------------------------

The recommended way to create tests is using the fluent API with ``scenario()``:

.. code-block:: python

   from giskard.checks import scenario, Conformity

   result = await (
       scenario("conversation_test")
       .interact("Hello", lambda inputs: generate_answer(inputs))
       .check(Conformity(key="interactions[-1].outputs", rule="response should be a friendly greeting"))
       .interact("Who invented the HTML?", lambda inputs: generate_answer(inputs))
       .check(Conformity(key="interactions[-1].outputs", rule="response should mention Tim Berners-Lee as the inventor of HTML"))
       .run()
   )
   print(f"Test passed: {result.passed}")

This creates a ``Scenario`` internally that executes interactions and checks sequentially. The fluent API is easier to read and write than manually constructing scenarios.


Advanced: TestCase and Scenario Classes
----------------------------------------

**Advanced Usage**: For programmatic test generation or advanced use cases, you can work with ``TestCase`` and ``Scenario`` classes directly:

.. code-block:: python

   from giskard.checks import TestCase, Scenario, InteractionSpec, from_fn

   # TestCase combines a trace with checks
   test_case = TestCase(
        trace=trace,
        checks=[check1, check2]
   )

   result = await test_case.run()

   # Scenario allows manual construction of sequences
   scenario = Scenario(
       name="my_scenario",
       sequence=[
           InteractionSpec(inputs="Hello", outputs=generate_answer),
           Conformity(key="interactions[-1].outputs", rule="response should be friendly"),
       ]
   )

   result = await scenario.run()

For most use cases, prefer the fluent API (``scenario().interact().check()``) as it's simpler and more readable.
