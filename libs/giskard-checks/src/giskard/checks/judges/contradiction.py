from typing import override

from giskard.agents import TemplateReference
from pydantic import Field
from pydantic.experimental.missing_sentinel import MISSING

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, provided_or_resolve
from .base import BaseLLMCheck


@Check.register("contradiction")
class Contradiction[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that fails only on clear contradictions with context.

    The check uses the same ``answer``/``answer_key`` and
    ``context``/``context_key`` inputs as the groundedness judge, but applies a
    permissive criterion: omissions and unsupported additions are tolerated
    unless they directly conflict with the reference context.
    """

    answer: str | MISSING = Field(
        default=MISSING, description="Input source for the answer to evaluate"
    )
    answer_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="Key to extract the answer from the trace",
    )
    context: str | list[str] | MISSING = Field(
        default=MISSING, description="Input source for the reference context"
    )
    context_key: JSONPathStr = Field(
        default="trace.last.metadata.context",
        description="Key to extract the context from the trace",
    )

    @override
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(
            template_name="giskard.checks::judges/contradiction.j2"
        )

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, str]:
        return {
            "answer": str(
                provided_or_resolve(trace, key=self.answer_key, value=self.answer)
            ),
            "context": str(
                provided_or_resolve(trace, key=self.context_key, value=self.context)
            ),
        }
