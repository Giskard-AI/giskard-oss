==========
Quickstart
==========

This guide will walk you through creating your first test with Giskard Checks in under 5 minutes.

.. note::
   All code examples in this documentation use async/await syntax. To run these examples, you can use ``asyncio.run()`` or run them within an async context (e.g., in a Jupyter notebook with async support, or in an async test framework like pytest-asyncio).


A simple example
----------------

Let's consider a simple question-answering bot. We want to test that the answers of our bot are correct according to some context information.

In the ``checks`` framework, all tests are performed on static representation of all data exchanged with the system under test (TODO: link to SUT). We call this a Trace (TODO: link to core concepts).

We call each turn of data exchange an Interaction. Think of an Interaction as a single API call to your system under test, with some inputs and some outputs.

.. note::
   For detailed explanations of these concepts, see :doc:`core-concepts`.

For our simple Q&A bot, we'll test a single interaction. The inputs and outputs can be anything the bot supports, as long as they are serializable to JSON. For now, we'll assume our bot takes an input string (question) and returns a string (the answer).

The easiest way to create a test is using the fluent API with ``scenario().interact().check()``:

.. code-block:: python

   from giskard.checks import scenario
   from giskard.checks.builtin import Groundedness

   # Define a simple bot function
   def qa_bot(question: str) -> str:
       # In practice, this would call your actual bot
       return "The capital of France is Paris."

   # Create and run a test scenario
   result = await (
       scenario("my_first_test")
       .interact(
           "What is the capital of France?",
           lambda inputs: qa_bot(inputs)
       )
       .check(Groundedness(
           name="answer is grounded",
           answer_key="interactions[-1].outputs",
           context="""France is a country in Western Europe. Its capital
                      and largest city is Paris, known for the Eiffel Tower
                      and the Louvre Museum."""
       ))
       .run()
   )
   print(f"Test passed: {result.passed}")

Note how we created the groundedness check:

- ``name``: this is an (optional) name for the check, to make it easier to interpret the results
- ``answer_key``: this is the key (in JSONPath) to the answer in the trace. In this case we want to check the ``outputs`` attribute of the last interaction (using ``interactions[-1].outputs``)
- ``context``: this is the context information that will be used to check if the answer is grounded. Note that a ``context_key`` is also available if we want to dynamically load the context from the trace itself (see next example).

TODO: result block, description


Structuring the interactions
----------------------------

As mentioned above, in practice the interaction inputs and outputs can take any form as long as they are serializable to JSON. For example, our bot could take input in the form of an OpenAI message object and return a structured output like this:

.. code-block:: json

   {
       "answer": "The capital of France is Paris.",
       "confidence": 0.93,
       "documents": [
            "France is a country in Western Europe. Its capital and largest city is Paris, known for the Eiffel Tower and the Louvre Museum.",
            "The Eiffel Tower is a wrought-iron lattice tower in Paris. It was completed in 1889."
        ]
   }

We can easily test this structured format using the fluent API:

.. code-block:: python

    from giskard.checks import scenario
    from giskard.checks.builtin import GreaterThan, Groundedness

    def structured_qa_bot(question: dict) -> dict:
        # Your bot that returns structured output
        return {
            "answer": "The capital of France is Paris.",
            "confidence": 0.93,
            "documents": [
                "France is a country in Western Europe. Its capital and largest city is Paris, known for the Eiffel Tower and the Louvre Museum.",
                "The Eiffel Tower is a wrought-iron lattice tower in Paris. It was completed in 1889."
            ]
        }

    result = await (
        scenario("structured_qa_test")
        .interact(
            {"role": "user", "content": "What is the capital of France?"},
            lambda inputs: structured_qa_bot(inputs)
        )
        .check(Groundedness(
            name="answer is grounded",
            answer_key="interactions[-1].outputs.answer",
            context_key="interactions[-1].outputs.documents",
        ))
        .check(GreaterThan(
            name="confidence is high",
            key="interactions[-1].outputs.confidence",
            threshold=0.90,
        ))
        .run()
    )
    print(f"Test passed: {result.passed}")

Note how this time we used ``context_key`` to obtain the context from the documents present in the trace itself. This is a common case for RAG systems. We also added a check to ensure the confidence is high.


Dynamic interactions
--------------------

In practice, we'll often want to create the outputs automatically from the system we are testing. The fluent API makes this easy - you can pass a function to generate outputs dynamically.

For example, our simple Q&A bot could be implemented using the OpenAI API:

.. code-block:: python

    from openai import OpenAI
    from giskard.checks import scenario
    from giskard.checks.builtin import Groundedness

    client = OpenAI()

    def get_answer(inputs: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": inputs}],
        )
        return response.choices[0].message.content

    # Use the function directly in the scenario
    result = await (
        scenario("dynamic_qa_test")
        .interact(
            "What is the capital of France?",
            lambda inputs: get_answer(inputs)
        )
        .check(Groundedness(
            name="answer is grounded",
            answer_key="interactions[-1].outputs",
            context="""France is a country in Western Europe. Its capital
                       and largest city is Paris, known for the Eiffel Tower
                       and the Louvre Museum."""
        ))
        .run()
    )
    print(f"Test passed: {result.passed}")

No need to specify outputs manually - they're generated automatically when the test runs!

Note that inputs can also be dynamically generated! This is especially useful when you are testing multi-turn scenarios. For example, you can generate the inputs based on the previous interactions.

Check out the :doc:`multi-turn` guide for more details on how to test multi-turn scenarios.
