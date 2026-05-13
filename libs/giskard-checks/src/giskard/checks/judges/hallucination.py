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
    """LLM-based check that detects fabricated facts in an answer.

    The check can evaluate an answer with or without a provided reference
    context. When context is provided, it is used as evidence for detecting
    fabricated facts, invented details, fake citations, and unsupported claims.

    Attributes
    ----------
    answer : str | None
        The answer text to evaluate.
    answer_key : str
        JSONPath expression to extract the answer from the trace
        (default: "trace.last.outputs").
    context : str | list[str] | None
        Optional reference context for the answer.
    context_key : str | None
        Optional JSONPath expression to extract context from the trace.
    """

    answer: str | None = Field(
        default=None, description="Input source for the answer to evaluate"
    )
    answer_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="Key to extract the answer from the trace",
    )
    context: str | list[str] | None = Field(
        default=None, description="Optional reference context for the answer"
    )
    context_key: JSONPathStr | None = Field(
        default=None,
        description="Optional key to extract reference context from the trace",
    )

    @override
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(
            template_name="giskard.checks::judges/hallucination.j2"
        )

    @override
    async def get_inputs(self, trace: TraceType) -> dict[str, str]:
        def fmt(value: object) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                return "\n\n".join(str(item) for item in value)
            return str(value)

        inputs = {
            "answer": fmt(
                provided_or_resolve(
                    trace,
                    key=self.answer_key,
                    value=provide_not_none(self.answer),
                )
            ),
            "context": "",
        }
        if self.context is not None or self.context_key is not None:
            inputs["context"] = fmt(
                provided_or_resolve(
                    trace,
                    key=self.context_key,
                    value=provide_not_none(self.context),
                )
            )
        return inputs
