import json

from giskard.agents.chat import Message
from giskard.agents.generators.base import BaseGenerator, GenerationParams, Response
from giskard.checks import CheckStatus, Conformity, Interaction, Trace
from pydantic import Field


class MockGenerator(BaseGenerator):
    passed: bool
    reason: str | None
    calls: list[list[Message]] = Field(default_factory=list)

    async def _complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        self.calls.append(messages)
        return Response(
            message=Message(
                role="assistant",
                content=json.dumps({"passed": self.passed, "reason": self.reason}),
            ),
            finish_reason="stop",
        )


async def test_run_returns_success() -> None:
    generator = MockGenerator(passed=True, reason="Rule is followed")
    conformity = Conformity(generator=generator, rule="The response must be polite.")
    result = await conformity.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Rule is followed"

    assert len(generator.calls) == 1
    # The prompt comes from the template file, so we check that the call was made
    assert len(generator.calls[0]) > 0


async def test_run_returns_failure() -> None:
    generator = MockGenerator(passed=False, reason="Rule is violated")
    conformity = Conformity(generator=generator, rule="The response must be polite.")
    result = await conformity.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.details["reason"] == "Rule is violated"

    assert len(generator.calls) == 1


async def test_rule_templating() -> None:
    generator = MockGenerator(passed=True, reason=None)
    conformity = Conformity(
        generator=generator,
        rule="The response should contain '{{ trace.interactions[-1].outputs.response }}'",
    )
    result = await conformity.run(
        Trace(
            interactions=[
                Interaction(inputs={"query": "Hello"}, outputs={"response": "Hello"})
            ]
        )
    )

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] is None

    assert len(generator.calls) == 1
    # Verify that the rule was templated correctly in the inputs
    # The formatted rule should contain "Hello" instead of the template placeholder
    assert "rule" in result.details["inputs"]
    assert "Hello" in result.details["inputs"]["rule"]


async def test_interaction_json_in_inputs() -> None:
    generator = MockGenerator(passed=True, reason=None)
    conformity = Conformity(generator=generator, rule="Test rule")
    interaction = Interaction(
        inputs={"query": "What is AI?"}, outputs={"response": "AI is..."}
    )
    result = await conformity.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert "inputs" in result.details
    assert "interaction" in result.details["inputs"]

    # Verify interaction is serialized as JSON
    interaction_json = result.details["inputs"]["interaction"]
    assert isinstance(interaction_json, str)
    parsed = json.loads(interaction_json)
    assert parsed["inputs"]["query"] == "What is AI?"
    assert parsed["outputs"]["response"] == "AI is..."


async def test_empty_interactions_uses_empty_json() -> None:
    generator = MockGenerator(passed=True, reason=None)
    conformity = Conformity(generator=generator, rule="Test rule")
    result = await conformity.run(Trace())

    assert result.status == CheckStatus.PASS
    assert "inputs" in result.details
    assert result.details["inputs"]["interaction"] == "{}"
