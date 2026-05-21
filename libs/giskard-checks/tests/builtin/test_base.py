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


class SequenceMockGenerator(BaseGenerator):
    responses: list[tuple[bool, str | None]]
    calls: list[Sequence[ChatMessage]] = Field(default_factory=list)

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        self.calls.append(messages)
        passed, reason = self.responses[len(self.calls) - 1]
        return CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(
                        content=json.dumps(
                            {
                                "passed": passed,
                                "reason": reason,
                            }
                        )
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )


class TestBaseLLMCheck:
    async def test_majority_consensus_returns_pass_with_run_details(self):
        class ConsensusCheck(BaseLLMCheck[str, str, Trace[str, str]]):
            @override
            def get_prompt(self) -> str:
                return "Evaluate."

        generator = SequenceMockGenerator(
            responses=[
                (True, "pass-1"),
                (False, "fail-2"),
                (True, "pass-3"),
            ]
        )
        check = ConsensusCheck(generator=generator, num_runs=3, consensus="majority")

        result = await check.run(Trace())

        assert result.passed
        assert result.message == "pass-1"
        assert result.details["consensus"] == "majority"
        assert result.details["num_runs"] == 3
        assert result.details["consensus_passed"] is True
        assert result.details["status_counts"] == {"pass": 2, "fail": 1}
        assert [run.status for run in result.details["runs"]] == [
            "pass",
            "fail",
            "pass",
        ]
        assert len(generator.calls) == 3

    async def test_unanimous_consensus_requires_all_runs_to_pass(self):
        class ConsensusCheck(BaseLLMCheck[str, str, Trace[str, str]]):
            @override
            def get_prompt(self) -> str:
                return "Evaluate."

        generator = SequenceMockGenerator(
            responses=[
                (True, "pass-1"),
                (False, "fail-2"),
                (True, "pass-3"),
            ]
        )
        check = ConsensusCheck(generator=generator, num_runs=3, consensus="unanimous")

        result = await check.run(Trace())

        assert result.failed
        assert result.message == "fail-2"
        assert result.details["consensus"] == "unanimous"
        assert result.details["consensus_passed"] is False

    async def test_any_consensus_passes_if_any_run_passes(self):
        class ConsensusCheck(BaseLLMCheck[str, str, Trace[str, str]]):
            @override
            def get_prompt(self) -> str:
                return "Evaluate."

        generator = SequenceMockGenerator(
            responses=[
                (False, "fail-1"),
                (False, "fail-2"),
                (True, "pass-3"),
            ]
        )
        check = ConsensusCheck(generator=generator, num_runs=3, consensus="any")

        result = await check.run(Trace())

        assert result.passed
        assert result.message == "pass-3"
        assert result.details["consensus"] == "any"
        assert result.details["consensus_passed"] is True

    async def test_single_run_keeps_default_behavior_unchanged(self):
        class ConsensusCheck(BaseLLMCheck[str, str, Trace[str, str]]):
            @override
            def get_prompt(self) -> str:
                return "Evaluate."

        generator = SequenceMockGenerator(responses=[(True, "pass-1")])
        check = ConsensusCheck(generator=generator)

        result = await check.run(Trace())

        assert result.passed
        assert result.details == {
            "reason": "pass-1",
            "inputs": {"trace": Trace()},
        }
        assert "runs" not in result.details

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
