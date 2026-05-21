from typing import Any, override

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
    """LLM-based check that validates faithful source representation.

    Uses an LLM to determine whether an answer accurately represents provided
    source material without distortion, misrepresentation, misleading selective
    omission, or unsupported claims.

    Attributes
    ----------
    answer : str | None
        The answer text to evaluate. When provided, takes priority over
        ``answer_key``.
    answer_key : JSONPathStr
        JSONPath expression to extract the answer from the trace
        (default: ``"trace.last.outputs"``).
    source : str | list[str] | None
        Source material the answer should faithfully represent. When provided,
        takes priority over ``source_key``.
    source_key : JSONPathStr | None
        Optional JSONPath expression to extract source material from the trace.
        If no direct source or source key is provided, the judge receives an
        empty source string.
    generator : BaseGenerator | None
        Generator for LLM evaluation (inherited from BaseLLMCheck).

    Examples
    --------
    >>> from giskard.agents.generators import Generator
    >>> check = Faithfulness(
    ...     answer="The report says revenue grew by 10%.",
    ...     source="The report states that revenue grew by 10%.",
    ...     generator=Generator(model="openai/gpt-4o")
    ... )
    """

    answer: str | None = Field(
        default=None,
        description="Input source for the answer to evaluate.",
    )
    answer_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="Key to extract the answer from the trace.",
    )
    source: str | list[str] | None = Field(
        default=None,
        description="Source material the answer should faithfully represent.",
    )
    source_key: JSONPathStr | None = Field(
        default=None,
        description="Optional key to extract the source material from the trace.",
    )

    @override
    def get_prompt(self) -> TemplateReference:
        """Return the bundled prompt template for faithfulness evaluation."""
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
            Template variables with ``answer`` and ``source`` keys.
        """
        answer = provided_or_resolve(
            trace,
            key=self.answer_key,
            value=provide_not_none(self.answer),
        )

        source: Any
        if self.source is not None:
            source = self.source
        elif self.source_key is not None:
            source = provided_or_resolve(trace, key=self.source_key)
        else:
            source = ""

        return {
            "answer": str(answer),
            "source": str(source),
        }
