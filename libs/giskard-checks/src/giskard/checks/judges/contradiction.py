from typing import cast, override

from giskard.agents.workflow import TemplateReference
from pydantic import Field, field_validator
from pydantic.experimental.missing_sentinel import MISSING

from ..core import Trace, extraction
from ..core.check import Check
from ..core.extraction import JSONPathStr
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

    answer: object = Field(
        default=MISSING, description="Input source for the answer to evaluate"
    )
    answer_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="Key to extract the answer from the trace",
    )
    context: object = Field(
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

    @field_validator("answer")
    @classmethod
    def _validate_answer(cls, value: object) -> object:
        if value is MISSING or isinstance(value, str):
            return value
        raise ValueError("answer must be a string")

    @field_validator("context")
    @classmethod
    def _validate_context(cls, value: object) -> object:
        if value is MISSING or isinstance(value, str):
            return value
        if isinstance(value, list):
            items = cast(list[object], value)
            if all(isinstance(item, str) for item in items):
                return items
        raise ValueError("context must be a string or a list of strings")

    @staticmethod
    def _stringify(value: object) -> str:
        return str(value)

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, str]:
        answer = cast(
            object,
            self.answer
            if self.answer is not MISSING
            else extraction.resolve(trace, self.answer_key),
        )
        context = cast(
            object,
            self.context
            if self.context is not MISSING
            else extraction.resolve(trace, self.context_key),
        )
        return {
            "answer": self._stringify(answer),
            "context": self._stringify(context),
        }
