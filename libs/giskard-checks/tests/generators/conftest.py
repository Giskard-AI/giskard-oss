import json
from collections.abc import Sequence
from typing import Any, override

from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import Interaction, Trace
from giskard.llm.types import AssistantMessage, ChatMessage, Choice, CompletionResponse
from pydantic import Field


class MockGenerator(BaseGenerator):
    """Shared mock generator for generator tests."""

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


class LLMTrace(Trace[Interaction[str, str]], frozen=True):
    """Shared minimal Trace implementation for generator tests."""

    def _repr_prompt_(self) -> str:
        if not self.interactions:
            return "**No interactions yet**"
        return "\n".join(
            f"[user]: {i.inputs}\n[assistant]: {i.outputs}" for i in self.interactions
        )
