==============
Custom Checks
==============

While Giskard Checks provides many built-in checks, you'll often need domain-specific validation. This guide shows you how to create custom checks tailored to your application.


Creating a Simple Check
------------------------

The easiest way to create a check is using ``from_fn``:

.. code-block:: python

   from giskard.checks import from_fn, Trace

   def my_validation(trace: Trace) -> bool:
       output = trace.interactions[-1].outputs
       return len(output) > 10

   check = from_fn(
       my_validation,
       name="min_length",
       success_message="Output meets minimum length",
       failure_message="Output too short"
   )

The function receives the trace and returns a boolean. ``from_fn`` wraps it into a proper ``Check`` instance.


Creating a Check Class
----------------------

For more control, create a custom check class:

.. code-block:: python

   from giskard.checks import Check, CheckResult, Trace

   @Check.register("min_length")
   class MinLengthCheck(Check):
       min_length: int = 10

       async def run(self, trace: Trace) -> CheckResult:
           output = trace.interactions[-1].outputs
           actual_length = len(output)

           if actual_length >= self.min_length:
               return CheckResult.success(
                   message=f"Output length {actual_length} meets minimum {self.min_length}"
               )

           return CheckResult.failure(
               message=f"Output length {actual_length} below minimum {self.min_length}"
           )

**Key Components:**

* ``@Check.register("kind")`` - Registers the check for polymorphic serialization
* Custom parameters as Pydantic fields (``min_length: int``)
* ``async def run()`` - Implements the check logic
* Return ``CheckResult.success()`` or ``CheckResult.failure()``


Using Your Custom Check
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from giskard.checks import InteractionSpec, TestCase

   check = MinLengthCheck(name="length_check", min_length=20)

   interaction = InteractionSpec(
       inputs="test",
       outputs="This is a reasonably long output"
   )

   tc = TestCase(interaction=interaction, checks=[check], name="test")
   result = await tc.run()


Adding Metrics
--------------

Checks can return numeric metrics for analysis:

.. code-block:: python

   from giskard.checks import Check, CheckResult, Trace

   @Check.register("readability")
   class ReadabilityCheck(Check):
       max_grade_level: int = 8

       async def run(self, trace: Trace) -> CheckResult:
           text = trace.interactions[-1].outputs

           # Calculate readability (Flesch-Kincaid grade level)
           grade_level = calculate_readability(text)

           if grade_level <= self.max_grade_level:
               return CheckResult.success(
                   message=f"Readability grade {grade_level:.1f} is acceptable",
                   metrics={"grade_level": grade_level}
               )

           return CheckResult.failure(
               message=f"Readability grade {grade_level:.1f} exceeds maximum {self.max_grade_level}",
               metrics={"grade_level": grade_level}
           )

   def calculate_readability(text: str) -> float:
       # Simplified readability calculation
       words = len(text.split())
       sentences = text.count('.') + text.count('!') + text.count('?')
       syllables = sum(count_syllables(word) for word in text.split())

       if sentences == 0:
           return 0

       return 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59

   def count_syllables(word: str) -> int:
       # Very simplified syllable counting
       return max(1, sum(1 for c in word.lower() if c in 'aeiou'))

The metrics can be used for tracking, analysis, and visualization:

.. code-block:: python

   result = await tc.run()
   for check_result in result.results:
       if check_result.metrics:
           print(f"{check_result.name}: {check_result.metrics}")


Extracting Values with JSONPath
--------------------------------

Use JSONPath to extract values from complex structures:

.. code-block:: python

   from giskard.checks import Check, CheckResult, Trace, JsonPathExtractor

   @Check.register("field_validator")
   class FieldValidatorCheck(Check):
       field_path: str
       min_value: float

       async def run(self, trace: Trace) -> CheckResult:
           extractor = JsonPathExtractor(key=self.field_path)
           value = extractor.extract(trace)

           if value >= self.min_value:
               return CheckResult.success(
                   message=f"Value {value} meets minimum {self.min_value}"
               )

           return CheckResult.failure(
               message=f"Value {value} below minimum {self.min_value}"
           )

Usage:

.. code-block:: python

   check = FieldValidatorCheck(
       name="confidence_check",
       field_path="interactions[-1].outputs.confidence",
       min_value=0.8
   )


Creating Custom LLM Checks
---------------------------

Build domain-specific LLM-based checks:

.. code-block:: python

   from pydantic import BaseModel
   from giskard.agents.workflow import TemplateReference
   from giskard.checks import BaseLLMCheck, Check, CheckResult, Trace

   class ToneEvaluation(BaseModel):
       is_professional: bool
       is_empathetic: bool
       tone_score: float
       reasoning: str
       passed: bool

   @Check.register("tone_check")
   class ToneCheck(BaseLLMCheck):
       required_tone: str = "professional and empathetic"

       def get_prompt(self) -> TemplateReference | str:
           return """
           Evaluate the tone of the assistant's response.

           User message: {{ inputs }}
           Assistant response: {{ outputs }}

           Required tone: {{ required_tone }}

           Provide:
           - is_professional: true/false
           - is_empathetic: true/false
           - tone_score: 0.0 to 1.0
           - reasoning: brief explanation
           - passed: true if tone matches requirements
           """

       @property
       def output_type(self) -> type[BaseModel]:
           return ToneEvaluation

       def get_inputs(self, trace: Trace) -> dict:
           return {
               "inputs": trace.interactions[-1].inputs,
               "outputs": trace.interactions[-1].outputs,
               "required_tone": self.required_tone
           }

       async def _handle_output(
           self,
           output_value: ToneEvaluation,
           template_inputs: dict,
           trace: Trace,
       ) -> CheckResult:
           if output_value.passed:
               return CheckResult.success(
                   message=f"Tone is appropriate: {output_value.reasoning}",
                   metrics={"tone_score": output_value.tone_score}
               )

           return CheckResult.failure(
               message=f"Tone issues: {output_value.reasoning}",
               metrics={"tone_score": output_value.tone_score},
               details={
                   "is_professional": output_value.is_professional,
                   "is_empathetic": output_value.is_empathetic
               }
           )


Async Checks
------------

Checks can be async for I/O operations:

.. code-block:: python

   import httpx
   from giskard.checks import Check, CheckResult, Trace

   @Check.register("api_validation")
   class APIValidationCheck(Check):
       api_endpoint: str

       async def run(self, trace: Trace) -> CheckResult:
           output = trace.interactions[-1].outputs

           # Make async HTTP request
           async with httpx.AsyncClient() as client:
               response = await client.post(
                   self.api_endpoint,
                   json={"text": output}
               )
               result = response.json()

           if result["is_valid"]:
               return CheckResult.success(
                   message="Output validated by external API",
                   details=result
               )

           return CheckResult.failure(
               message=f"Validation failed: {result.get('error')}",
               details=result
           )


Stateful Checks
---------------

Checks can maintain state across scenarios (use with caution):

.. code-block:: python

   from giskard.checks import Check, CheckResult, Trace

   @Check.register("consistency_tracker")
   class ConsistencyTracker(Check):
       def __init__(self, **data):
           super().__init__(**data)
           self.seen_values = set()

       async def run(self, trace: Trace) -> CheckResult:
           output = trace.interactions[-1].outputs

           if output in self.seen_values:
               return CheckResult.failure(
                   message=f"Duplicate output detected: {output}"
               )

           self.seen_values.add(output)
           return CheckResult.success(
               message="Output is unique",
               metrics={"unique_count": len(self.seen_values)}
           )

**Note:** Stateful checks can make tests harder to reason about. Consider passing state through the trace instead when possible.


Composing Checks
----------------

Build complex checks by composing simpler ones:

.. code-block:: python

   from giskard.checks import Check, CheckResult, Trace

   @Check.register("composite_quality")
   class CompositeQualityCheck(Check):
       min_length: int = 10
       max_length: int = 1000
       required_keywords: list[str] = []

       async def run(self, trace: Trace) -> CheckResult:
           output = trace.interactions[-1].outputs
           issues = []

           # Length checks
           if len(output) < self.min_length:
               issues.append(f"Too short (minimum {self.min_length})")
           if len(output) > self.max_length:
               issues.append(f"Too long (maximum {self.max_length})")

           # Keyword checks
           missing_keywords = [
               kw for kw in self.required_keywords
               if kw.lower() not in output.lower()
           ]
           if missing_keywords:
               issues.append(f"Missing keywords: {', '.join(missing_keywords)}")

           if issues:
               return CheckResult.failure(
                   message="; ".join(issues),
                   details={"issues": issues}
               )

           return CheckResult.success(
               message="All quality checks passed",
               metrics={
                   "length": len(output),
                   "keywords_found": len(self.required_keywords) - len(missing_keywords)
               }
           )


Domain-Specific Examples
------------------------

Medical Transcript Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from giskard.checks import Check, CheckResult, Trace

   @Check.register("medical_transcript")
   class MedicalTranscriptCheck(Check):
       required_sections: list[str] = [
           "Chief Complaint",
           "History of Present Illness",
           "Assessment",
           "Plan"
       ]

       async def run(self, trace: Trace) -> CheckResult:
           transcript = trace.interactions[-1].outputs

           missing_sections = [
               section for section in self.required_sections
               if section not in transcript
           ]

           if missing_sections:
               return CheckResult.failure(
                   message=f"Missing required sections: {', '.join(missing_sections)}",
                   details={"missing": missing_sections}
               )

           return CheckResult.success(
               message="All required sections present"
           )

Financial Report Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import re
   from giskard.checks import Check, CheckResult, Trace

   @Check.register("financial_report")
   class FinancialReportCheck(Check):
       require_disclaimer: bool = True
       allow_predictions: bool = False

       async def run(self, trace: Trace) -> CheckResult:
           report = trace.interactions[-1].outputs
           issues = []

           # Check for required disclaimer
           if self.require_disclaimer:
               disclaimer_patterns = [
                   r"not financial advice",
                   r"for informational purposes only",
                   r"consult.*financial advisor"
               ]
               if not any(re.search(p, report, re.IGNORECASE) for p in disclaimer_patterns):
                   issues.append("Missing required financial disclaimer")

           # Check for unauthorized predictions
           if not self.allow_predictions:
               prediction_patterns = [
                   r"will (increase|decrease|rise|fall)",
                   r"expect.*to (grow|decline)",
                   r"predicted? (gain|loss)"
               ]
               if any(re.search(p, report, re.IGNORECASE) for p in prediction_patterns):
                   issues.append("Contains unauthorized predictions")

           if issues:
               return CheckResult.failure(
                   message="; ".join(issues),
                   details={"violations": issues}
               )

           return CheckResult.success(message="Report meets compliance requirements")


Testing Custom Checks
----------------------

Write tests for your custom checks:

.. code-block:: python

   import pytest
   from giskard.checks import Interaction, Trace

   @pytest.mark.asyncio
   async def test_min_length_check():
       # Test passing case
       trace_pass = Trace(interactions=[
           Interaction(inputs="test", outputs="This is long enough")
       ])

       check = MinLengthCheck(name="test", min_length=10)
       result = await check.run(trace_pass)

       assert result.passed
       assert "meets minimum" in result.message

       # Test failing case
       trace_fail = Trace(interactions=[
           Interaction(inputs="test", outputs="Short")
       ])

       result = await check.run(trace_fail)

       assert not result.passed
       assert "below minimum" in result.message


Best Practices
--------------

**1. Make Checks Focused**

Each check should validate one specific aspect:

.. code-block:: python

   # Good: Focused checks
   LengthCheck(min_length=10)
   ToneCheck(required_tone="professional")
   FormatCheck(expected_format="json")

   # Avoid: One check doing too much
   MegaCheck(validate_everything=True)

**2. Provide Clear Messages**

Messages should explain what passed or failed:

.. code-block:: python

   # Good
   return CheckResult.failure(
       message="Confidence score 0.65 is below required threshold 0.8"
   )

   # Avoid
   return CheckResult.failure(message="Check failed")

**3. Use Type Hints**

Leverage Pydantic's type validation:

.. code-block:: python

   @Check.register("typed_check")
   class TypedCheck(Check):
       threshold: float  # Validated as float
       keywords: list[str]  # Validated as list of strings
       enabled: bool = True  # Optional with default

**4. Add Documentation**

Document your checks:

.. code-block:: python

   @Check.register("documented_check")
   class DocumentedCheck(Check):
       """Validates that outputs meet quality standards.

       This check evaluates:
       - Minimum length requirements
       - Presence of required keywords
       - Readability score

       Parameters
       ----------
       min_length : int
           Minimum character count
       required_keywords : list[str]
           Keywords that must appear
       """

       min_length: int
       required_keywords: list[str]


Next Steps
----------

* Apply custom checks in :doc:`../tutorials/index`
* Review :doc:`single-turn` and :doc:`multi-turn` for usage patterns
* See the :doc:`core-concepts` for architecture details
