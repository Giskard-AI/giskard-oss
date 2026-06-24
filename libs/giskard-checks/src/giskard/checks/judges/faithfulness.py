from typing import Any, override

from giskard.agents.workflow import TemplateReference
from giskard.core import provide_not_none
from pydantic import BaseModel, Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, provided_or_resolve, resolve
from ..core.result import CheckResult, CheckStatus, Metric
from .base import BaseLLMCheck


class FaithfulnessCheckResult(BaseModel):
    """Structured output returned by the faithfulness judge."""

    score: float = Field(
        ..., ge=0.0, le=1.0, description="Faithfulness score between 0 and 1."
    )
    passed: bool = Field(..., description="Whether the answer is faithful.")
    reason: str | None = Field(
        default=None, description="Optional explanation for the result."
    )


@Check.register("faithfulness")
class Faithfulness[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that evaluates whether an answer faithfully represents source material."""

    answer: str | None = Field(
        default=None,
        description="Answer to evaluate. If None, extracted from the trace using answer_key.",
    )
    answer_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the answer from the trace.",
    )
    source: str | list[str] | None = Field(
        default=None,
        description="Source material the answer should faithfully represent.",
    )
    source_key: JSONPathStr | None = Field(
        default=None,
        description="JSONPath expression to extract source material from the trace.",
    )
    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum faithfulness score required to pass.",
    )

    @property
    @override
    def output_type(self) -> type[BaseModel]:
        return FaithfulnessCheckResult

    @override
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(template_name="giskard.checks::judges/faithfulness.j2")

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, Any]:
        answer = provided_or_resolve(
            trace,
            key=self.answer_key,
            value=provide_not_none(self.answer),
        )
        if answer is None or isinstance(answer, NoMatch):
            raise ValueError(
                f"Could not resolve answer from trace using key '{self.answer_key}' "
                "and no direct answer was provided."
            )

        source = self._resolve_source(trace)
        if source is None or isinstance(source, NoMatch):
            raise ValueError(
                f"Could not resolve source from trace using key '{self.source_key}' "
                "and no direct source was provided."
            )

        return {
            "answer": answer,
            "source": self._format_source(source),
            "threshold": self.threshold,
        }

    def _resolve_source(
        self, trace: Trace[InputType, OutputType]
    ) -> str | list[str] | NoMatch:
        if self.source is not None:
            return self.source

        if self.source_key is None:
            return NoMatch(key="source")

        return resolve(trace, self.source_key)

    @staticmethod
    def _format_source(source: Any) -> str:
        if isinstance(source, list):
            return "\n\n".join(str(item) for item in source)
        return str(source)

    @override
    async def _handle_output(
        self,
        output_value: BaseModel,
        template_inputs: dict[str, Any],
        trace: TraceType,
    ) -> CheckResult:
        _ = trace
        if not isinstance(output_value, FaithfulnessCheckResult):
            raise NotImplementedError(
                f"Custom output type {type(output_value)} requires overriding _handle_output"
            )

        score = output_value.score
        passed = output_value.passed and score >= self.threshold
        details = {
            "reason": output_value.reason,
            "score": score,
            "threshold": self.threshold,
            "passed": output_value.passed,
            "inputs": template_inputs,
        }
        metric = Metric(name="faithfulness", value=score)

        if passed:
            return CheckResult(
                status=CheckStatus.PASS,
                message=output_value.reason
                or f"Faithfulness score {score:.2f} meets threshold {self.threshold:.2f}.",
                metrics=[metric],
                details=details,
            )

        if score < self.threshold:
            message = (
                output_value.reason
                or f"Faithfulness score {score:.2f} is below threshold {self.threshold:.2f}."
            )
        else:
            message = output_value.reason or "The answer is not faithful to the source."

        return CheckResult(
            status=CheckStatus.FAIL,
            message=message,
            metrics=[metric],
            details=details,
        )
