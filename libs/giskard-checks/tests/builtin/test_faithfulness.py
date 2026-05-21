"""Unit tests for the Faithfulness check."""

import json
from collections.abc import Sequence
from typing import Any, cast, override

from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import (
    Check,
    CheckResult,
    CheckStatus,
    Faithfulness,
    Interaction,
    Trace,
)
from giskard.llm.types import AssistantMessage, ChatMessage, Choice, CompletionResponse
from pydantic import Field

_EXPECTED_INPUT_KEYS = frozenset({"answer", "source"})


def _assert_faithfulness_inputs(result: CheckResult) -> dict[str, str]:
    """Assert every Faithfulness run records full template inputs."""
    assert "inputs" in result.details
    inputs = result.details["inputs"]
    assert isinstance(inputs, dict)
    assert set(inputs.keys()) == _EXPECTED_INPUT_KEYS
    return cast(dict[str, str], inputs)


@BaseGenerator.register("mock_faithfulness")
class MockGenerator(BaseGenerator):
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


def serialization_roundtrip[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    check: Faithfulness[InputType, OutputType, TraceType],
) -> Faithfulness[InputType, OutputType, TraceType]:
    roundtrip_check = Check.model_validate(check.model_dump())
    assert isinstance(roundtrip_check, Faithfulness)
    return cast(Faithfulness[InputType, OutputType, TraceType], roundtrip_check)


async def test_faithful_answer_passes() -> None:
    generator = MockGenerator(
        passed=True,
        reason="The answer preserves the source meaning.",
    )
    check = Faithfulness(
        generator=generator,
        answer="The report says revenue grew by 10%.",
        source="The report states that revenue grew by 10%.",
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "The answer preserves the source meaning."
    inputs = _assert_faithfulness_inputs(result)
    assert inputs["answer"] == "The report says revenue grew by 10%."
    assert inputs["source"] == "The report states that revenue grew by 10%."
    assert len(generator.calls) == 1


async def test_distorted_answer_fails() -> None:
    generator = MockGenerator(
        passed=False,
        reason="The answer changes revenue growth into a decline.",
    )
    check = Faithfulness(
        generator=generator,
        answer="The report says revenue declined by 10%.",
        source="The report states that revenue grew by 10%.",
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert (
        result.details["reason"] == "The answer changes revenue growth into a decline."
    )
    inputs = _assert_faithfulness_inputs(result)
    assert inputs["answer"] == "The report says revenue declined by 10%."
    assert inputs["source"] == "The report states that revenue grew by 10%."
    assert len(generator.calls) == 1


async def test_partial_faithfulness_with_material_distortion_fails() -> None:
    generator = MockGenerator(
        passed=False,
        reason="The answer is partly accurate but removes a material caveat.",
    )
    check = Faithfulness(
        generator=generator,
        answer=(
            "The clinical trial showed the treatment was effective for all patients."
        ),
        source=(
            "The clinical trial showed improvement for adults in the study, "
            "but results for children were inconclusive."
        ),
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["reason"] == (
        "The answer is partly accurate but removes a material caveat."
    )
    inputs = _assert_faithfulness_inputs(result)
    assert "effective for all patients" in inputs["answer"]
    assert "results for children were inconclusive" in inputs["source"]


async def test_direct_answer_and_source_override_trace_values() -> None:
    generator = MockGenerator(passed=True)
    check = Faithfulness(
        generator=generator,
        answer="Direct answer",
        source="Direct source",
        answer_key="trace.last.outputs.answer",
        source_key="trace.last.metadata.source",
    )
    interaction = Interaction(
        inputs="Summarize",
        outputs={"answer": "Trace answer"},
        metadata={"source": "Trace source"},
    )

    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    inputs = _assert_faithfulness_inputs(result)
    assert inputs["answer"] == "Direct answer"
    assert inputs["source"] == "Direct source"


async def test_source_key_extracts_from_metadata() -> None:
    generator = MockGenerator(passed=True)
    check = Faithfulness(
        generator=generator,
        answer_key="trace.last.outputs.summary",
        source_key="trace.last.metadata.source",
    )
    interaction = Interaction(
        inputs="Summarize the document",
        outputs={"summary": "The policy applies to EU customers."},
        metadata={"source": "The policy applies to customers located in the EU."},
    )

    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    inputs = _assert_faithfulness_inputs(result)
    assert inputs["answer"] == "The policy applies to EU customers."
    assert inputs["source"] == "The policy applies to customers located in the EU."


async def test_custom_source_key_extracts_nested_value() -> None:
    generator = MockGenerator(passed=True)
    check = Faithfulness(
        generator=generator,
        answer_key="trace.interactions[0].outputs.answer",
        source_key="trace.interactions[0].metadata.documents[0].text",
    )
    interaction = Interaction(
        inputs={"question": "What changed?"},
        outputs={"answer": "The service window starts at 09:00."},
        metadata={"documents": [{"text": "The service window starts at 09:00."}]},
    )

    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    inputs = _assert_faithfulness_inputs(result)
    assert inputs["answer"] == "The service window starts at 09:00."
    assert inputs["source"] == "The service window starts at 09:00."


async def test_source_accepts_string_and_list_values() -> None:
    string_generator = MockGenerator(passed=True)
    string_check = Faithfulness(
        generator=string_generator,
        answer="A direct summary.",
        source="A direct source.",
    )
    string_result = await string_check.run(Trace())

    assert string_result.status == CheckStatus.PASS
    string_inputs = _assert_faithfulness_inputs(string_result)
    assert string_inputs["source"] == "A direct source."

    list_generator = MockGenerator(passed=True)
    list_check = Faithfulness(
        generator=list_generator,
        answer="The docs mention alpha and beta.",
        source=["The docs mention alpha.", "The docs mention beta."],
    )
    list_result = await list_check.run(Trace())

    assert list_result.status == CheckStatus.PASS
    list_inputs = _assert_faithfulness_inputs(list_result)
    assert "The docs mention alpha." in list_inputs["source"]
    assert "The docs mention beta." in list_inputs["source"]


async def test_empty_trace_and_unset_source_do_not_crash() -> None:
    generator = MockGenerator(passed=False, reason="No source material was supplied.")
    check = Faithfulness(generator=generator)

    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    inputs = _assert_faithfulness_inputs(result)
    assert inputs["answer"] == "No match for key: trace.last.outputs"
    assert inputs["source"] == ""


async def test_missing_source_key_does_not_crash() -> None:
    generator = MockGenerator(passed=False, reason="Source key did not resolve.")
    check = Faithfulness(
        generator=generator,
        answer="Some answer",
        source_key="trace.last.metadata.source",
    )
    interaction = Interaction(inputs="Question", outputs="Answer")

    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.FAIL
    inputs = _assert_faithfulness_inputs(result)
    assert inputs["answer"] == "Some answer"
    assert inputs["source"] == "No match for key: trace.last.metadata.source"


async def test_serialization_roundtrip() -> None:
    generator = MockGenerator(passed=True, reason="Roundtrip works.")
    check = Faithfulness(
        generator=generator,
        answer_key="trace.last.outputs.answer",
        source_key="trace.last.metadata.source",
    )
    roundtrip_check = serialization_roundtrip(check)
    interaction = Interaction(
        inputs="Summarize",
        outputs={"answer": "The source says the launch is in May."},
        metadata={"source": "The source says the launch is in May."},
    )

    result = await roundtrip_check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Roundtrip works."
    inputs = _assert_faithfulness_inputs(result)
    assert inputs["answer"] == "The source says the launch is in May."
    assert inputs["source"] == "The source says the launch is in May."
    assert isinstance(roundtrip_check.generator, MockGenerator)
    assert len(roundtrip_check.generator.calls) == 1


def test_public_imports() -> None:
    from giskard.checks import Faithfulness as TopLevelFaithfulness
    from giskard.checks.builtin import Faithfulness as BuiltinFaithfulness
    from giskard.checks.judges import Faithfulness as JudgeFaithfulness

    assert TopLevelFaithfulness is Faithfulness
    assert BuiltinFaithfulness is Faithfulness
    assert JudgeFaithfulness is Faithfulness
