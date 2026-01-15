from typing import Any

import pytest
from giskard import agents
from giskard.agents.chat import Message
from giskard.agents.generators import BaseGenerator, GenerationParams
from giskard.agents.generators.base import Response
from pydantic import BaseModel, Field, PrivateAttr, ValidationError


class DummyOutputModel(BaseModel):
    """Test output model for validation testing."""

    name: str = Field(description="A name field")
    score: int = Field(description="A score between 0 and 100", ge=0, le=100)
    active: bool = Field(description="Whether the item is active")


class MockValidationGenerator(BaseGenerator):
    """Mock generator that returns predefined responses to test validation."""

    responses: list[Any] = Field(default_factory=list)
    _call_count: int = PrivateAttr(default=0)

    async def _complete(
        self,
        messages: list[Message],
        params: GenerationParams | None = None,
    ) -> Response:
        if self._call_count >= len(self.responses):
            # If we run out of responses, return the last one
            response_content = (
                self.responses[-1]
                if self.responses
                else '{"name": "default", "score": 50, "active": true}'
            )
        else:
            response_content = self.responses[self._call_count]

        self._call_count += 1

        return Response(
            message=Message(role="assistant", content=response_content),
            finish_reason="stop",
        )


async def test_output_model_strict_validation_success():
    """Test that valid JSON output passes strict validation."""
    valid_json = '{"name": "test", "score": 85, "active": true}'

    generator = MockValidationGenerator(responses=[valid_json])
    workflow = agents.ChatWorkflow(generator=generator)

    chat = await (
        workflow.chat("Please provide a response", role="user")
        .with_output(DummyOutputModel, strict=True, num_retries=2)
        .run()
    )

    # Should succeed without errors
    assert chat.last.content == valid_json
    output = chat.output
    assert isinstance(output, DummyOutputModel)
    assert output.name == "test"
    assert output.score == 85
    assert output.active is True


async def test_output_model_strict_validation_failure():
    """Test that invalid JSON output raises ValidationError in strict mode."""
    invalid_json = '{"name": "test", "score": 150, "active": "not_boolean"}'  # score > 100, active not boolean

    generator = MockValidationGenerator(responses=[invalid_json])
    workflow = agents.ChatWorkflow(generator=generator)

    with pytest.raises(agents.errors.WorkflowError) as exc_info:
        await (
            workflow.chat("Please provide a response", role="user")
            .with_output(DummyOutputModel, strict=True, num_retries=0)  # No retries
            .run()
        )
    assert isinstance(exc_info.value.exception, ValidationError)


async def test_output_model_strict_validation_fails_when_no_content():
    """Test that invalid JSON output raises ValidationError in strict mode."""
    generator = MockValidationGenerator(responses=[None])
    workflow = agents.ChatWorkflow(generator=generator)

    with pytest.raises(agents.errors.WorkflowError) as exc_info:
        await (
            workflow.chat("Please provide a response", role="user")
            .with_output(DummyOutputModel, strict=True, num_retries=0)  # No retries
            .run()
        )
    assert isinstance(exc_info.value.exception, ValidationError)


async def test_output_model_retry_success():
    """Test that retry mechanism works when first response is invalid but second is valid."""
    invalid_json = (
        '{"name": "test", "score": 150, "active": "invalid"}'  # Invalid first response
    )
    valid_json = (
        '{"name": "test", "score": 85, "active": true}'  # Valid second response
    )

    generator = MockValidationGenerator(responses=[invalid_json, valid_json])
    workflow = agents.ChatWorkflow(generator=generator)

    chat = await (
        workflow.chat("Please provide a response", role="user")
        .with_output(DummyOutputModel, strict=True, num_retries=2)
        .run()
    )

    # Should succeed after retry
    assert chat.last.content == valid_json
    output = chat.output
    assert isinstance(output, DummyOutputModel)
    assert output.name == "test"
    assert output.score == 85
    assert output.active is True

    # Verify the generator was called twice
    assert generator._call_count == 2


async def test_output_model_retry_exhausted():
    """Test that ValidationError is raised after all retries are exhausted."""
    invalid_json_1 = '{"name": "test", "score": 150, "active": true}'  # score too high
    invalid_json_2 = '{"name": "test", "score": -10, "active": true}'  # score too low
    invalid_json_3 = (
        '{"name": "test", "score": 50, "active": "invalid"}'  # active not boolean
    )

    generator = MockValidationGenerator(
        responses=[invalid_json_1, invalid_json_2, invalid_json_3]
    )
    workflow = agents.ChatWorkflow(generator=generator)

    with pytest.raises(agents.errors.WorkflowError) as exc_info:
        await (
            workflow.chat("Please provide a response", role="user")
            .with_output(
                DummyOutputModel, strict=True, num_retries=2
            )  # 3 total attempts
            .run()
        )

    # Should have tried all 3 attempts
    assert generator._call_count == 3
    # The original exception should be a ValidationError
    assert isinstance(exc_info.value.exception, ValidationError)


async def test_output_model_non_strict_mode():
    """Test that non-strict mode doesn't validate output and doesn't retry."""
    invalid_json = '{"name": "test", "score": 150, "active": "invalid"}'

    generator = MockValidationGenerator(responses=[invalid_json])
    workflow = agents.ChatWorkflow(generator=generator)

    chat = await (
        workflow.chat("Please provide a response", role="user")
        .with_output(DummyOutputModel, strict=False, num_retries=2)
        .run()
    )

    # Should succeed without validation
    assert chat.last.content == invalid_json
    # Only one call should have been made (no retries)
    assert generator._call_count == 1

    # Parsing will fail when we try to access .output
    with pytest.raises(ValidationError):
        chat.output


async def test_output_model_no_output_model_set():
    """Test that without output_model, strict validation is not applied."""
    invalid_json = '{"invalid": "json", "that": "would", "fail": "validation"}'

    generator = MockValidationGenerator(responses=[invalid_json])
    workflow = agents.ChatWorkflow(generator=generator)

    chat = await workflow.chat("Please provide a response", role="user").run()

    # Should succeed without any validation
    assert chat.last.content == invalid_json
    assert generator._call_count == 1


async def test_output_model_custom_retry_count():
    """Test that custom retry count is respected."""
    invalid_responses = [
        '{"name": "test", "score": 150, "active": true}',  # attempt 1
        '{"name": "test", "score": 151, "active": true}',  # attempt 2
        '{"name": "test", "score": 152, "active": true}',  # attempt 3
        '{"name": "test", "score": 153, "active": true}',  # attempt 4
        '{"name": "test", "score": 154, "active": true}',  # attempt 5
    ]

    generator = MockValidationGenerator(responses=invalid_responses)
    workflow = agents.ChatWorkflow(generator=generator)

    with pytest.raises(agents.errors.WorkflowError):
        await (
            workflow.chat("Please provide a response", role="user")
            .with_output(
                DummyOutputModel, strict=True, num_retries=4
            )  # 5 total attempts
            .run()
        )

    # Should have made exactly 5 attempts (1 + 4 retries)
    assert generator._call_count == 5


async def test_output_model_zero_retries():
    """Test that setting num_retries=0 means no retries, just one attempt."""
    invalid_json = '{"name": "test", "score": 150, "active": true}'

    generator = MockValidationGenerator(responses=[invalid_json])
    workflow = agents.ChatWorkflow(generator=generator)

    with pytest.raises(agents.errors.WorkflowError):
        await (
            workflow.chat("Please provide a response", role="user")
            .with_output(DummyOutputModel, strict=True, num_retries=0)
            .run()
        )

    # Should have made exactly 1 attempt
    assert generator._call_count == 1
