from unittest.mock import AsyncMock

import pytest
from giskard.agents.chat import Message
from giskard.agents.generators.base import BaseGenerator, GenerationParams, Response
from giskard.agents.generators.mixins import WithRetryPolicy
from giskard.agents.generators.retry import RetryPolicy


class RetriableError(BaseException):
    """A retriable error."""


class MockGenerator(WithRetryPolicy, BaseGenerator):
    """A mock generator for testing the WithRetryPolicy mixin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._complete_mock = AsyncMock()

    def _should_retry(self, err: Exception) -> bool:
        return isinstance(err, RetriableError)

    async def _complete_once(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        return await self._complete_mock(messages, params)


@pytest.fixture
def mock_response(self):
    """Create a mock response."""
    return Response(
        message=Message(role="assistant", content="Test response"),
        finish_reason="stop",
    )


def test_with_retries_helper_method():
    # Generator with no default policy
    generator = MockGenerator()
    new_generator = generator.with_retries(max_retries=3)
    assert new_generator.retry_policy is not None
    assert new_generator.retry_policy.max_retries == 3
    assert new_generator.retry_policy.base_delay == 1.0  # default

    # Generator with existing policy
    generator = MockGenerator(retry_policy=RetryPolicy(max_retries=2, base_delay=0.5))
    new_generator = generator.with_retries(max_retries=5)
    assert new_generator.retry_policy is not None
    assert new_generator.retry_policy.max_retries == 5
    assert new_generator.retry_policy.base_delay == 0.5

    new_generator = generator.with_retries(4, base_delay=10)
    assert new_generator.retry_policy.max_retries == 4
    assert new_generator.retry_policy.base_delay == 10


async def test_raises_exception_after_retries_exhausted():
    generator = MockGenerator(
        retry_policy=RetryPolicy(max_retries=3, base_delay=1e-3),
    )
    generator._complete_mock.side_effect = RetriableError("Test error")

    with pytest.raises(RetriableError):
        await generator.complete(
            messages=[Message(role="user", content="Test message")]
        )

    assert generator._complete_mock.call_count == 3


async def test_raises_exception_if_not_retriable():
    generator = MockGenerator(
        retry_policy=RetryPolicy(max_retries=3, base_delay=1e-3),
    )
    generator._complete_mock.side_effect = ValueError("Test error")

    with pytest.raises(ValueError):
        await generator.complete(
            messages=[Message(role="user", content="Test message")]
        )

    assert generator._complete_mock.call_count == 1


async def test_retries_with_result():
    generator = MockGenerator(
        retry_policy=RetryPolicy(max_retries=3, base_delay=1e-3),
    )
    generator._complete_mock.side_effect = [
        RetriableError("Test error"),
        RetriableError("Test error"),
        Response(
            message=Message(role="assistant", content="Test response"),
            finish_reason="stop",
        ),
    ]

    res = await generator.complete(
        messages=[Message(role="user", content="Test message")]
    )
    assert res.message.content == "Test response"
    assert res.finish_reason == "stop"

    assert generator._complete_mock.call_count == 3


async def test_retries_works_with_batch_complete():
    generator = MockGenerator(
        retry_policy=RetryPolicy(max_retries=3, base_delay=1e-3),
    )
    generator._complete_mock.side_effect = [
        RetriableError("Test error"),
        RetriableError("Test error"),
        Response(
            message=Message(role="assistant", content="Test response"),
            finish_reason="stop",
        ),
    ]

    res = await generator.batch_complete(
        messages=[
            [Message(role="user", content="Test message")],
        ]
    )

    assert len(res) == 1
    assert res[0].message.content == "Test response"
    assert res[0].finish_reason == "stop"

    assert generator._complete_mock.call_count == 3
