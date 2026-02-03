from typing import override

from giskard.agents.workflow import TemplateReference
from giskard.core import provide_not_none
from pydantic import Field

from ..core.check import Check
from ..core.extraction import provided_or_resolve
from ..core.trace import Trace
from .base import BaseLLMCheck


@Check.register("groundedness")
class Groundedness[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that validates answers are grounded in context.

    Uses an LLM to determine if an answer is properly supported by
    the provided context documents.

    Attributes
    ----------
    answer : str | None
        The answer text to evaluate for groundedness.
    answer_key : str
        JSONPath expression to extract the answer from the trace
        (default: "trace.last.outputs").

        Can use `trace.last` (preferred) or `trace.interactions[-1]` for JSONPath expressions.
    context : list[str] | None
        List of context documents that should support the answer.
    context_key : str
        JSONPath expression to extract the context from the trace
        (default: "trace.last.metadata.context").

        Can use `trace.last` (preferred) or `trace.interactions[-1]` for JSONPath expressions.
    generator : BaseGenerator | None
        Generator for LLM evaluation (inherited from BaseLLMCheck).

    Examples
    --------
    >>> from giskard.agents.generators import Generator
    >>> check = Groundedness(
    ...     answer="The Eiffel Tower is in Paris.",
    ...     context=["Paris is the capital of France.", "It's located in Europe."],
    ...     generator=Generator(model="openai/gpt-4o")
    ... )
    """

    answer: str | None = Field(
        default=None, description="Input source for the answer to evaluate"
    )
    answer_key: str = Field(
        default="trace.last.outputs",
        description="Key to extract the answer from the trace",
    )
    context: str | list[str] | None = Field(
        default=None, description="Input source for the reference context"
    )
    context_key: str = Field(
        default="trace.last.metadata.context",
        description="Key to extract the context from the trace",
    )

    @override
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(template_name="giskard.checks::checks/groundedness.j2")

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, str]:
        """Build template variables from resolved inputs.

        Parameters
        ----------
        trace : Trace
            Trace for resolving inputs.

        Returns
        -------
        dict[str, str]
            Template variables with 'answer' and 'context' keys.
        """
        return {
            "answer": str(
                provided_or_resolve(
                    trace,
                    key=self.answer_key,
                    value=provide_not_none(self.answer),
                )
            ),
            "context": str(
                provided_or_resolve(
                    trace,
                    key=self.context_key,
                    value=provide_not_none(self.context),
                )
            ),
        }
