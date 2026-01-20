=============
Core Concepts
=============

Understanding the key concepts in Giskard Checks will help you write effective tests for your AI applications.


Overview
--------

Giskard Checks is built around a few core primitives that work together:

TODO

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

An ``InteractionSpec`` describes *how* to generate an interaction. Both inputs and outputs can be generated dynamically:

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

This is very common when you are testing multi-turn scenarios, where inputs and outputs are generated based on previous interactions. See TODO for practical examples.

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


TestCase
--------

A ``TestCase`` is combines a trace and the checks that will be applied to it.

.. code-block:: python

   from giskard.checks import TestCase, InteractionSpec, from_fn

   test_case = TestCase(
        trace=trace,
        checks=[check1, check2]
   )

   result = await test_case.run()

This will give us a result object with the results of the checks.


Scenario
--------

A ``Scenario`` allows you to compose multiple steps of testing in sequences alternating some interactions and some checks. This is useful when you are testing complex multi-turn scenarios and you want to test intermediate checkpoints.

For example, we can test a simple conversation flow with two turns:

.. code-block:: python

   from giskard.checks import Scenario, InteractionSpec, from_fn

   scenario = Scenario.from_sequence(
        InteractionSpec(inputs="Hello", outputs=generate_answer),
        Conformity(key="last.outputs", rule="response should be a friendly greeting"),
        InteractionSpec(inputs="Who invented the HTML?", outputs=generate_answer),
        Conformity(key="last.outputs", rule="response should mention Tim Berners-Lee as the inventor of HTML"),
   )

   result = await scenario.run()
