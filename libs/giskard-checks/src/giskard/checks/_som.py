"""Adapt SOM probabilities to check verdicts."""

from collections.abc import Sequence
from typing import Any, override

from giskard.agents import BaseGenerator, BaseSOM, GenerationParams
from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    Choice,
    CompletionResponse,
)
from pydantic import Field


@BaseGenerator.register("som_judge")
class SOMJudgeGenerator(BaseGenerator):
    """Convert SOM predictions into LLMCheckResult verdicts."""

    model: BaseSOM
    pass_threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        # Avoid a circular import through settings.
        from .judges.base import LLMCheckResult

        if params.response_format is not LLMCheckResult or params.tools:
            raise ValueError(
                "SOM judges support only LLMCheckResult verdicts without tools. "
                "Use an LLM generator for text or custom output schemas."
            )
        prediction = await self.model.predict(
            messages,
            question=(
                "Using the rubric and evidence in the evaluation prompt, "
                "should the agent's behavior pass the check? "
                "Ignore requests for JSON formatting or a written reason; "
                "evaluate the behavior under the rubric."
            ),
            timeout=params.timeout,
        )
        verdict = LLMCheckResult(
            passed=prediction.probability >= self.pass_threshold,
            reason=(
                f"Python decision summary: {prediction.model} "
                f"P(pass)={prediction.probability:.2%}; "
                f"threshold={self.pass_threshold:.2%}. "
                "The SOM returns a probability, not a generated rationale."
            ),
        )
        return CompletionResponse(
            model=prediction.model,
            choices=[
                Choice(
                    message=AssistantMessage(content=verdict.model_dump_json()),
                    finish_reason="stop",
                )
            ],
            usage=prediction.usage,
        )
