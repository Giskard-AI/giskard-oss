==============
Built-in Checks
==============

Ready-to-use checks for common testing scenarios.

.. currentmodule:: giskard.checks


Function-Based Checks
---------------------

from_fn
~~~~~~~

.. autofunction:: from_fn

Create a check from a simple function.

**Example:**

.. code-block:: python

   from giskard.checks import from_fn, Trace

   def my_validation(trace: Trace) -> bool:
       return len(trace.interactions[-1].outputs) > 10

   check = from_fn(
       my_validation,
       name="min_length",
       success_message="Output is long enough",
       failure_message="Output too short"
   )


FnCheck
~~~~~~~

.. autoclass:: FnCheck
   :members:
   :undoc-members:
   :show-inheritance:

Check class created by ``from_fn``.


String Matching
---------------

StringMatchingCheck
~~~~~~~~~~~~~~~~~~~

.. autoclass:: StringMatchingCheck
   :members:
   :undoc-members:
   :show-inheritance:

Check if text contains, starts with, ends with, or exactly matches a pattern.

**Example:**

.. code-block:: python

   from giskard.checks import StringMatchingCheck

   check = StringMatchingCheck(
       name="contains_answer",
       content="Paris",
       key="interactions[-1].outputs.answer",
       evaluation_mode="contains"  # or "exact", "starts_with", "ends_with"
   )


Equality Checks
---------------

EqualityCheck
~~~~~~~~~~~~~

.. autoclass:: EqualityCheck
   :members:
   :undoc-members:
   :show-inheritance:

Check if extracted value equals expected value.

**Example:**

.. code-block:: python

   from giskard.checks import EqualityCheck

   check = EqualityCheck(
       name="correct_confidence",
       expected=0.95,
       key="interactions[-1].outputs.confidence"
   )


Extraction Checks
-----------------

ExtractionCheck
~~~~~~~~~~~~~~~

.. autoclass:: ExtractionCheck
   :members:
   :undoc-members:
   :show-inheritance:

Base class for checks that extract values from traces using JSONPath.


LLM-Based Checks
----------------

BaseLLMCheck
~~~~~~~~~~~~

.. autoclass:: BaseLLMCheck
   :members:
   :undoc-members:
   :show-inheritance:

Base class for checks that use LLMs for evaluation.


Groundedness
~~~~~~~~~~~~

.. autoclass:: Groundedness
   :members:
   :undoc-members:
   :show-inheritance:

Check if outputs are grounded in the provided context/inputs.

**Example:**

.. code-block:: python

   from giskard.checks import Groundedness

   check = Groundedness(
       name="answer_grounded",
       description="Verify answer is based on context"
   )


Conformity
~~~~~~~~~~

.. autoclass:: Conformity
   :members:
   :undoc-members:
   :show-inheritance:

Check if outputs conform to instructions or specifications.

**Example:**

.. code-block:: python

   from giskard.checks import Conformity

   check = Conformity(
       name="follows_instructions",
       description="Ensure response follows the given instructions"
   )


LLMJudge
~~~~~~~~

.. autoclass:: LLMJudge
   :members:
   :undoc-members:
   :show-inheritance:

Custom LLM-based evaluation with user-defined prompt.

**Example:**

.. code-block:: python

   from giskard.checks import LLMJudge

   check = LLMJudge(
       name="tone_check",
       prompt="""
       Evaluate if the response has a professional tone.

       Input: {{ inputs }}
       Output: {{ outputs }}

       Return 'passed: true' if professional, 'passed: false' otherwise.
       """
   )


LLMCheckResult
~~~~~~~~~~~~~~

.. autoclass:: LLMCheckResult
   :members:
   :undoc-members:
   :show-inheritance:

Result type for LLM-based checks with structured output.


Custom Check Examples
---------------------

Creating a custom LLM check:

.. code-block:: python

   from pydantic import BaseModel
   from giskard.agents.workflow import TemplateReference
   from giskard.checks import BaseLLMCheck, Check, CheckResult, Trace

   class CustomEvaluation(BaseModel):
       score: float
       passed: bool
       reasoning: str

   @Check.register("custom_eval")
   class CustomEvalCheck(BaseLLMCheck):
       threshold: float = 0.8

       def get_prompt(self) -> TemplateReference | str:
           return "Evaluate this: {{ outputs }}"

       @property
       def output_type(self) -> type[BaseModel]:
           return CustomEvaluation

       async def _handle_output(
           self,
           output_value: CustomEvaluation,
           template_inputs: dict,
           trace: Trace,
       ) -> CheckResult:
           if output_value.score >= self.threshold:
               return CheckResult.success(
                   message=f"Score {output_value.score} meets threshold"
               )
           return CheckResult.failure(
               message=f"Score {output_value.score} below threshold"
           )
