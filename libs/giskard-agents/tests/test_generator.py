import time
from unittest.mock import patch

import pytest
from giskard.agents.chat import Chat, Message
from giskard.agents.generators.base import GenerationParams, Response
from giskard.agents.generators.litellm_generator import LiteLLMGenerator
from giskard.agents.rate_limiter import RateLimiter, _rate_limiters
from giskard.agents.templates import MessageTemplate
from giskard.agents.workflow import ChatWorkflow
from litellm import ModelResponse


@pytest.fixture
def mock_response():
    return ModelResponse(
        choices=[
            dict(
                finish_reason="stop",
                message=dict(role="assistant", content="Mock response"),
            )
        ]
    )


async def test_litellm_generator_completion_with_mock(
    generator: LiteLLMGenerator, mock_response
):
    with patch(
        "giskard.agents.generators.litellm_generator.acompletion",
        return_value=mock_response,
    ):
        response = await generator.complete(
            messages=[Message(role="user", content="Test message")]
        )

        assert response.message.role == "assistant"
        assert response.message.content == "Mock response"
        assert response.finish_reason == "stop"


async def test_generator_completion(generator: LiteLLMGenerator):
    response = await generator.complete(
        messages=[
            Message(
                role="system",
                content="You are a helpful assistant, greeting the user with 'Hello I am TestBot'.",
            ),
            Message(role="user", content="Hello, world!"),
        ]
    )

    assert isinstance(response, Response)
    assert response.message.role == "assistant"
    assert "I am TestBot" in response.message.content
    assert response.finish_reason == "stop"


async def test_generator_chat(generator: LiteLLMGenerator):
    test_message = "Hello, world!"
    pipeline = generator.chat(test_message)

    assert isinstance(pipeline, ChatWorkflow)
    assert len(pipeline.messages) == 1
    assert isinstance(pipeline.messages[0], MessageTemplate)
    assert pipeline.messages[0].role == "user"
    assert pipeline.messages[0].content_template == test_message

    chat = await pipeline.run()

    assert isinstance(chat, Chat)

    chats = await pipeline.run_many(3)

    assert len(chats) == 3
    assert isinstance(chats[0], Chat)
    assert isinstance(chats[1], Chat)
    assert isinstance(chats[2], Chat)


async def test_litellm_generator_gets_rate_limiter(mock_response):
    rate_limiter = RateLimiter.from_rpm(rpm=60, max_concurrent=1)
    generator = LiteLLMGenerator(model="test-model", rate_limiter=rate_limiter)
    with patch(
        "giskard.agents.generators.litellm_generator.acompletion",
        return_value=mock_response,
    ):
        start_time = time.monotonic()
        for _ in range(3):
            await generator.complete(
                messages=[Message(role="user", content="Test message")]
            )
        end_time = time.monotonic()

    # Distribution of request:
    # t = 0.0 -> request 1
    # t = 1.0 -> request 2
    # t = 2.0 -> request 3
    elapsed_time = end_time - start_time
    assert elapsed_time >= 2
    assert elapsed_time < 2 + rate_limiter.strategy.min_interval


async def test_generator_without_rate_limiter(mock_response):
    generator = LiteLLMGenerator(model="test-model")
    with patch(
        "giskard.agents.generators.litellm_generator.acompletion",
        return_value=mock_response,
    ):
        start_time = time.monotonic()
        for _ in range(3):
            await generator.complete(
                messages=[Message(role="user", content="Test message")]
            )
        end_time = time.monotonic()

    elapsed_time = end_time - start_time
    assert elapsed_time < 10e-3  # arbitrary small number, here 10ms


async def test_generator_rate_limiter_context():
    rate_limiter = RateLimiter.from_rpm(
        rpm=100, rate_limiter_id="test_generator_rate_limiter_context"
    )
    generator = LiteLLMGenerator(
        model="test-model", rate_limiter="test_generator_rate_limiter_context"
    )
    assert generator.rate_limiter is rate_limiter


def test_generator_with_params():
    generator = LiteLLMGenerator(model="test-model")
    generator = generator.with_params(temperature=0.5)
    assert generator.params.temperature == 0.5

    new_generator = generator.with_params(temperature=0.7)
    assert new_generator.params.temperature == 0.7
    assert generator.params.temperature == 0.5

    int_generator = new_generator.with_params(response_format=int)
    assert int_generator.params.response_format is int
    assert new_generator.params.response_format is None
    assert generator.params.response_format is None


def test_generator_with_params_and_rate_limiter():
    """Test that with_params works correctly with a rate limiter."""
    rate_limiter = RateLimiter.from_rpm(rpm=100, max_concurrent=5)
    generator = LiteLLMGenerator(model="test-model", rate_limiter=rate_limiter)

    # Verify initial state
    assert generator.rate_limiter is rate_limiter

    # Call with_params and verify rate limiter is preserved
    generator_with_params = generator.with_params(temperature=0.5, max_tokens=100)
    assert generator_with_params.params.temperature == 0.5
    assert generator_with_params.params.max_tokens == 100
    # Verify rate limiter is preserved and the same instance
    assert generator_with_params.rate_limiter is rate_limiter

    # Verify original generator is unchanged
    assert generator.params.temperature == 1.0  # default value
    assert generator.params.max_tokens is None
    assert generator.rate_limiter is rate_limiter


def test_generator_serialization_keep_rate_limiter_instance():
    """Test that serializing and deserializing a generator preserves the rate limiter instance."""
    rate_limiter = RateLimiter.from_rpm(rpm=100, max_concurrent=5)
    generator = LiteLLMGenerator(model="test-model", rate_limiter=rate_limiter)

    json_str = generator.model_dump_json()
    deserialized_generator = LiteLLMGenerator.model_validate_json(json_str)

    assert deserialized_generator.rate_limiter is rate_limiter


def test_generator_serialization_recreate_rate_limiter_instance_if_not_in_registry():
    """Test that deserializing a generator recreates the rate limiter if it's not in the registry."""
    rate_limiter = RateLimiter.from_rpm(rpm=100, max_concurrent=5)
    generator = LiteLLMGenerator(model="test-model", rate_limiter=rate_limiter)

    json_str = generator.model_dump_json()
    del _rate_limiters[rate_limiter.rate_limiter_id]
    deserialized_generator = LiteLLMGenerator.model_validate_json(json_str)

    assert deserialized_generator.rate_limiter is not rate_limiter
    assert (
        deserialized_generator.rate_limiter.rate_limiter_id
        == rate_limiter.rate_limiter_id
    )
    assert (
        deserialized_generator.rate_limiter.strategy.min_interval
        == rate_limiter.strategy.min_interval
    )
    assert (
        deserialized_generator.rate_limiter.strategy.max_concurrent
        == rate_limiter.strategy.max_concurrent
    )


async def test_generator_with_params_overwrite(mock_response):
    # ARRANGE: Create a generator with base parameters.
    generator = LiteLLMGenerator(model="test-model").with_params(
        temperature=0.5,  # This should be preserved.
        max_tokens=100,  # This should be overwritten.
    )

    with patch(
        "giskard.agents.generators.litellm_generator.acompletion",
        return_value=mock_response,
    ) as mock_acompletion:
        # ACT: Call complete() with overriding parameters.
        await generator.complete(
            messages=[Message(role="user", content="Test message")],
            params=GenerationParams(max_tokens=200),
        )

        # ASSERT: Verify that parameters were merged correctly.
        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args.kwargs
        assert (
            call_kwargs["temperature"] == 0.5
        )  # Preserved from the generator's params.
        assert (
            call_kwargs["max_tokens"] == 200
        )  # Overwritten by the complete() call's params.
        assert call_kwargs["model"] == "test-model"
