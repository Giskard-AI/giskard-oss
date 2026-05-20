import json
from collections.abc import Sequence
from typing import Any, override

import pytest
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import BaseLLMCheck, Trace
from giskard.llm.types import AssistantMessage, ChatMessage, Choice, CompletionResponse
from pydantic import BaseModel, Field


class MockGenerator(BaseGenerator):
    score: float
    passed: bool
    reasoning: str
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
                            {
                                "score": self.score,
                                "passed": self.passed,
                                "reasoning": self.reasoning,
                            }
                        )
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )


class TestBaseLLMCheck:
    async def test_custom_output_type_requires_handle_output(self):
        class CustomOutputType(BaseModel):
            score: float
            passed: bool
            reasoning: str

        class CustomLLMCheck(BaseLLMCheck[str, str, Trace[str, str]]):
            @override
            def get_prompt(self) -> str:
                return "What is the score?"

            @property
            @override
            def output_type(self) -> type[BaseModel]:
                return CustomOutputType

        generator = MockGenerator(score=0.85, passed=True, reasoning="Good score")
        check = CustomLLMCheck(generator=generator)
        with pytest.raises(NotImplementedError):
            _ = await check.run(Trace())
