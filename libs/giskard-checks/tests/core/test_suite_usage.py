import json
from collections.abc import Sequence
from typing import Any, override

import pytest
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import Equals, LLMJudge, Scenario, Suite
from giskard.checks.core.result import SuiteResult
from giskard.checks.core.suite_usage import SuiteUsage
from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    Choice,
    CompletionResponse,
    Usage,
)
from rich.console import Console


class UsageRecordingGenerator(BaseGenerator):
    """Mock generator that returns fixed usage on every completion."""

    prompt_tokens: int = 100
    completion_tokens: int = 40
    calls: int = 0

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        _ = messages, params, metadata
        self.calls += 1
        return CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(
                        content=json.dumps({"passed": True, "reason": "ok"})
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            usage=Usage(
                input_tokens=self.prompt_tokens,
                output_tokens=self.completion_tokens,
                total_tokens=self.prompt_tokens + self.completion_tokens,
            ),
        )


@pytest.mark.asyncio
async def test_suite_aggregates_llm_token_usage():
    generator = UsageRecordingGenerator()
    judge = LLMJudge(
        name="usage_judge",
        generator=generator,
        prompt="Did the assistant respond appropriately?",
    )
    suite = Suite(name="usage_suite")
    suite.append(
        Scenario("llm_scenario").interact(inputs="hello", outputs="hello").check(judge)
    )
    suite.append(
        Scenario("string_scenario")
        .interact(inputs="world", outputs="world")
        .check(Equals(expected_value="world", key="trace.last.outputs"))
    )

    result = await suite.run(verbose=False)

    assert result.usage == SuiteUsage(
        prompt_tokens=100,
        completion_tokens=40,
        total_tokens=140,
    )
    assert generator.calls == 1


@pytest.mark.asyncio
async def test_non_llm_suite_usage_stays_zero():
    suite = Suite(name="plain_suite")
    suite.append(
        Scenario("equals_only")
        .interact(inputs="ping", outputs="ping")
        .check(Equals(expected_value="ping", key="trace.last.outputs"))
    )

    result = await suite.run(verbose=False)

    assert result.usage == SuiteUsage()
    assert result.usage.total_tokens == 0


def test_print_report_omits_usage_footer_when_zero():
    result = SuiteResult(results=[], duration_ms=0)
    console = Console(width=120, record=True)
    result.print_report(console=console)
    output = console.export_text()

    assert "tokens:" not in output


def test_print_report_includes_usage_footer_when_nonzero():
    result = SuiteResult(
        results=[],
        duration_ms=0,
        usage=SuiteUsage(
            prompt_tokens=128_400,
            completion_tokens=39_200,
            total_tokens=167_600,
        ),
    )
    console = Console(width=120, record=True)
    result.print_report(console=console)
    output = console.export_text()

    assert "tokens: in=128400  out=39200  total=167600" in output
