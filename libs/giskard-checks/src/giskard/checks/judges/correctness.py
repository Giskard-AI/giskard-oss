from typing import Any, override

from giskard.agents.workflow import TemplateReference
from giskard.core import provide_not_none
from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, provided_or_resolve
from .base import BaseLLMCheck


@Check.register("correctness")
class Correctness[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that validates an answer against a reference (ground-truth).

    Uses an LLM to decide whether the agent's answer is correct relative to a
    reference answer. The judge prompt receives the same ``Trace`` instance that is
    passed to ``run`` as the "conversation": the full interaction
    history, not a separate string field.

    **How the trace appears in the prompt**

    Template rendering (via the agents Jinja environment) formats values for the LLM.
    If the trace type implements ``_repr_prompt_()`` (see
    ``giskard.agents.templates.LLMFormattable``), that method supplies the
    conversation text. Otherwise the trace
    is serialized as indented JSON from ``model_dump()`` (Pydantic).

    Attributes
    ----------
    description : str | None
        Description of the agent under test.
    description_key : str
        JSONPath expression to extract the description from the trace
        (default: ``trace.annotations.description``).

        Can use ``trace.last`` (preferred) or ``trace.interactions[-1]`` for JSONPath expressions.
    answer : str | None
        The agent's answer to evaluate (typically the response to the final user message).
    answer_key : str
        JSONPath expression to extract the answer from the trace
        (default: ``trace.last.outputs``).
    reference_answer : str | None
        Reference answer representing the correct response.
    reference_answer_key : str
        JSONPath expression to extract the reference answer from the trace
        (default: ``trace.last.metadata.reference_answer``).
    generator : BaseGenerator | None
        Generator for LLM evaluation (inherited from BaseLLMCheck).

    Examples
    --------
    >>> from giskard.agents.generators import Generator
    >>> from giskard.checks import Correctness, Interaction, Trace
    >>> trace = Trace(
    ...     annotations={"description": "A factual Q&A assistant"},
    ...     interactions=[
    ...         Interaction(
    ...             inputs="Capital of France?",
    ...             outputs="Paris.",
    ...             metadata={"reference_answer": "The capital of France is Paris."},
    ...         )
    ...     ],
    ... )
    >>> check = Correctness(generator=Generator(model="openai/gpt-4o"))
    >>> # await check.run(trace)
    """

    description: str | None = Field(
        default=None, description="Description of the agent under test"
    )
    description_key: JSONPathStr = Field(
        default="trace.annotations.description",
        description="Key to extract the agent description from the trace",
    )
    answer: str | None = Field(
        default=None, description="Agent answer to evaluate for correctness"
    )
    answer_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="Key to extract the answer from the trace",
    )
    reference_answer: str | None = Field(
        default=None, description="Reference (ground-truth) answer"
    )
    reference_answer_key: JSONPathStr = Field(
        default="trace.last.metadata.reference_answer",
        description="Key to extract the reference answer from the trace",
    )

    @override
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(template_name="giskard.checks::judges/correctness.j2")

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, Any]:
        """Build template variables from the trace and explicit field overrides.

        The ``conversation`` variable in the judge template is the ``trace`` object
        itself (formatted for the LLM as described in the class docstring).

        Parameters
        ----------
        trace : Trace
            Trace passed to ``run``; also supplied as template input ``conversation``.

        Returns
        -------
        dict[str, Any]
            Template variables: ``description`` and ``answer`` and ``reference_answer``
            as strings; ``conversation`` as the trace instance.
        """
        return {
            "description": str(
                provided_or_resolve(
                    trace,
                    key=self.description_key,
                    value=provide_not_none(self.description),
                )
            ),
            "conversation": trace,
            "answer": str(
                provided_or_resolve(
                    trace,
                    key=self.answer_key,
                    value=provide_not_none(self.answer),
                )
            ),
            "reference_answer": str(
                provided_or_resolve(
                    trace,
                    key=self.reference_answer_key,
                    value=provide_not_none(self.reference_answer),
                )
            ),
        }
