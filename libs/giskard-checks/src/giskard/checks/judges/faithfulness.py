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
    """LLM-based check that validates answers faithfully represent source material.

    Uses an LLM to determine if an answer accurately represents the source
    without distortion, misrepresentation, or selective quoting.

    Distinct from Groundedness — Groundedness checks if claims are supported
    by context, while Faithfulness checks if the answer accurately represents
    the source material holistically (including distortion and misrepresentation).

    Attributes
    ----------
    answer : str | None
        The answer text to evaluate for faithfulness.
    answer_key : str
        JSONPath expression to extract the answer from the trace
        (default: "trace.last.outputs").

        Can use `trace.last` (preferred) or `trace.interactions[-1]` for JSONPath expressions.
    source : str | list[str] | None
        Source material that the answer should faithfully represent.
    source_key : JSONPathStr | None
        JSONPath expression to extract the source from the trace
        (default: "trace.last.metadata.source").

        Can use `trace.last` (preferred) or `trace.interactions[-1]` for JSONPath expressions.
    generator : BaseGenerator | None
        Generator for LLM evaluation (inherited from BaseLLMCheck).

    Examples
    --------
    >>> from giskard.agents.generators import Generator
    >>> check = Faithfulness(
    ...     answer="The document says climate change is not a concern.",
    ...     source=["The document states climate change is a serious concern requiring action."],
    ...     generator=Generator(model="openai/gpt-4o")
    ... )
    """

    answer: str | None = Field(
        default=None, description="Input source for the answer to evaluate"
    )
    answer_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="Key to extract the answer from the trace",
    )
    source: str | list[str] | None = Field(
        default=None, description="Input source material the answer should faithfully represent"
    )
    source_key: JSONPathStr = Field(
        default="trace.last.metadata.source",
        description="Key to extract the source material from the trace",
    )

    @override
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(template_name="giskard.checks::judges/faithfulness.j2")

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
            Template variables with 'answer' and 'source' keys.
        """
        return {
            "answer": str(
                provided_or_resolve(
                    trace,
                    key=self.answer_key,
                    value=provide_not_none(self.answer),
                )
            ),
            "source": str(
                provided_or_resolve(
                    trace,
                    key=self.source_key,
                    value=provide_not_none(self.source),
                )
            ),
        }