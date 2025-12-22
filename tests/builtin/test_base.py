import json
from typing import override

import pytest
from giskard.agents.chat import Message
from giskard.agents.generators.base import BaseGenerator, GenerationParams, Response
from giskard.checks import BaseLLMCheck, Trace
from pydantic import BaseModel, Field


class MockGenerator(BaseGenerator):
    score: float
    passed: bool
    reasoning: str
    calls: list[list[Message]] = Field(default_factory=list)

    @override
    async def _complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        self.calls.append(messages)
        return Response(
            message=Message(
                role="assistant",
                content=json.dumps(
                    {
                        "score": self.score,
                        "passed": self.passed,
                        "reasoning": self.reasoning,
                    }
                ),
            ),
            finish_reason="stop",
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
            def output_type(self) -> type[BaseModel]:
                return CustomOutputType

        generator = MockGenerator(score=0.85, passed=True, reasoning="Good score")
        check = CustomLLMCheck(generator=generator)
        with pytest.raises(NotImplementedError):
            await check.run(Trace())
