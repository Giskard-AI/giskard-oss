from typing import override

from giskard.agents.workflow import TemplateReference
from giskard.core import provide_not_none
from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, provided_or_resolve
from .base import BaseLLMCheck


@Check.register("hallucination")
class Hallucination[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that detects fabricated facts in model outputs.

    Hallucination is distinct from Groundedness: Groundedness checks if claims
    are supported by context, while Hallucination specifically detects fabricated
    information and can operate with or without reference context.

    Attributes
    ----------
    answer : str | None
        The answer to evaluate for hallucinations. If None, extracted from the
        trace using ``answer_key``.
    answer_key : JSONPathStr
        JSONPath expression to extract the answer from the trace
        (default: ``"trace.last.outputs"``).
    context : str | list[str] | None
        Optional reference context. When provided, takes priority over
        ``context_key``. The check can run without context (detecting
        internally inconsistent or implausible fabrications).
    context_key : JSONPathStr
        JSONPath expression to extract the context from the trace
        (default: ``"trace.last.metadata.context"``).

    Examples
    --------
    >>> from giskard.checks import Hallucination, Scenario
    >>> scenario = (
    ...     Scenario(name="no_hallucination")
    ...     .interact(
    ...         inputs="What year was Python created?",
    ...         outputs="Python was created in 1991 by Guido van Rossum.",
    ...         metadata={"context": "Python was first released in 1991."}
    ...     )
    ...     .check(Hallucination())
    ... )
    """

    answer: str | None = Field(
        default=None,
        description="The answer to evaluate for hallucinations. If None, extracted from the trace using answer_key.",
    )
    answer_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the answer from the trace.",
    )
    context: str | list[str] | None = Field(
        default=None,
        description="Optional reference context. Takes priority over context_key.",
    )
    context_key: JSONPathStr = Field(
        default="trace.last.metadata.context",
        description="JSONPath expression to extract the context from the trace.",
    )

    @override
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(
            template_name="giskard.checks::judges/hallucination.j2"
        )

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, str]:
        answer = str(
            provided_or_resolve(
                trace,
                key=self.answer_key,
                value=provide_not_none(self.answer),
            )
        )

        if self.context is not None:
            resolved_context = self.context
        else:
            resolved_context = provided_or_resolve(trace, key=self.context_key)

        context = (
            "\n\n".join(map(str, resolved_context))
            if isinstance(resolved_context, list)
            else str(resolved_context)
        )

        return {
            "answer": answer,
            "context": context,
        }
