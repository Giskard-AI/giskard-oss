import json
from collections.abc import Sequence
from typing import Any, override

from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import Trace
from giskard.llm.types import AssistantMessage, ChatMessage, Choice, CompletionResponse
from pydantic import Field


class MockGenerator(BaseGenerator):
    """Generic mock generator: returns pre-canned JSON responses from a list."""

    responses: list[dict[str, Any]]
    index: int = 0
    calls: list[Sequence[ChatMessage]] = Field(default_factory=list)

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        self.calls.append(messages)
        message = AssistantMessage(content=json.dumps(self.responses[self.index]))
        self.index += 1
        return CompletionResponse(
            choices=[Choice(message=message, finish_reason="stop", index=0)]
        )


class MockJudgeGenerator(BaseGenerator):
    """Mock generator that returns a pre-configured judge verdict (passed/reason)."""

    passed: bool
    reason: str | None = None
    calls: list[Sequence[ChatMessage]] = Field(default_factory=list)

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        self.calls.append(messages)
        return CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(
                        content=json.dumps(
                            {"passed": self.passed, "reason": self.reason}
                        )
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )


class LLMTrace(Trace[str, str], frozen=True):
    """Minimal Trace implementation for tests."""

    def _repr_prompt_(self) -> str:
        if not self.interactions:
            return "**No interactions yet**"
        return "\n".join(
            f"[user]: {i.inputs}\n[assistant]: {i.outputs}" for i in self.interactions
        )
