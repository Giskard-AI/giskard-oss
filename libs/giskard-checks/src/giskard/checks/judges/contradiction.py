from typing import override

from giskard.agents import TemplateReference
from pydantic import Field
from pydantic.experimental.missing_sentinel import MISSING

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, provided_or_resolve
from ..core.result import CheckResult
from ._inputs import error_if_unresolved_answer_or_context
from .base import BaseLLMCheck, format_prompt_text


@Check.register("contradiction")
class Contradiction[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that fails only on clear contradictions with context.

    The check uses the same ``answer``/``target_key`` and
    ``context``/``context_key`` inputs as the groundedness judge, but applies a
    permissive criterion: omissions and unsupported additions are tolerated
    unless they directly conflict with the reference context.

    Attributes
    ----------
    answer : str | MISSING
        The answer text to evaluate. If omitted, extracted from the trace using
        ``target_key``.
    target_key : JSONPathStr
        JSONPath expression to extract the answer from the trace
        (default: ``"trace.last.outputs"``).
    """

    answer: str | MISSING = Field(
        default=MISSING, description="Input source for the answer to evaluate"
    )
    target_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description=("Key to extract the answer from the trace."),
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
    async def run(self, trace: TraceType) -> CheckResult:
        """Return ERROR when answer/context keys do not resolve; else run the judge."""
        if early := error_if_unresolved_answer_or_context(
            trace,
            answer=self.answer,
            answer_key=self.target_key,
            context=self.context,
            context_key=self.context_key,
        ):
            return early
        return await super().run(trace)

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, str]:
        return {
            "answer": format_prompt_text(
                provided_or_resolve(trace, key=self.target_key, value=self.answer)
            ),
            "context": format_prompt_text(
                provided_or_resolve(trace, key=self.context_key, value=self.context)
            ),
        }
