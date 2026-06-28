from typing import override

from giskard.agents.workflow import TemplateReference
from giskard.core import provide_not_none
from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, provided_or_resolve
from .base import BaseLLMCheck


@Check.register("faithfulness")
class Faithfulness[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that evaluates whether the answer faithfully represents its source.

    Faithfulness is distinct from Groundedness: Groundedness checks if every claim
    is supported by context, while Faithfulness assesses whether the answer accurately
    represents the source without distortion, misrepresentation, or selective quoting.

    Attributes
    ----------
    answer : str | None
        The answer to evaluate. If None, extracted from the trace using ``answer_key``.
    answer_key : JSONPathStr
        JSONPath expression to extract the answer from the trace
        (default: ``"trace.last.outputs"``).
    source : str | list[str] | None
        The source material to evaluate faithfulness against. When provided,
        takes priority over ``source_key``.
    source_key : JSONPathStr
        JSONPath expression to extract the source from the trace
        (default: ``"trace.last.metadata.context"``).

    Examples
    --------
    >>> from giskard.checks import Faithfulness, Scenario
    >>> scenario = (
    ...     Scenario(name="faithful_answer")
    ...     .interact(
    ...         inputs="Summarize this document",
    ...         outputs="The document states that Python was created in 1991.",
    ...         metadata={"context": "Python was first released in 1991 by Guido van Rossum."}
    ...     )
    ...     .check(Faithfulness())
    ... )
    """

    answer: str | None = Field(
        default=None,
        description="The answer to evaluate for faithfulness. If None, extracted from the trace using answer_key.",
    )
    answer_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the answer from the trace.",
    )
    source: str | list[str] | None = Field(
        default=None,
        description="Source material to evaluate faithfulness against. Takes priority over source_key.",
    )
    source_key: JSONPathStr = Field(
        default="trace.last.metadata.context",
        description="JSONPath expression to extract the source material from the trace.",
    )

    @override
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(template_name="giskard.checks::judges/faithfulness.j2")

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, str]:
        answer = str(
            provided_or_resolve(
                trace,
                key=self.answer_key,
                value=provide_not_none(self.answer),
            )
        )
        resolved_source = provided_or_resolve(
            trace,
            key=self.source_key,
            value=provide_not_none(self.source),
        )
        source = (
            "\n\n".join(map(str, resolved_source))
            if isinstance(resolved_source, list)
            else str(resolved_source)
        )

        return {
            "answer": answer,
            "source": source,
        }
