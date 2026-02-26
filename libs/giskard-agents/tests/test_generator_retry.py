from unittest.mock import AsyncMock

import pytest
import tenacity as t
from giskard.agents.chat import Message
from giskard.agents.generators import RetryPolicy, WithRetryPolicy
from giskard.agents.generators.base import BaseGenerator, GenerationParams, Response


class RetriableError(Exception):
    """A retriable error."""


class MockGenerator(WithRetryPolicy, BaseGenerator):
    """A mock generator for testing the WithRetryPolicy mixin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._complete_mock = AsyncMock()
        self._sleep_times: list[float] = []

    def _should_retry(self, err: Exception) -> bool:
        return isinstance(err, RetriableError)

    async def _attempt_complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        return await self._complete_mock(messages, params)

    def _tenacity_before_sleep(self, retry_state: t.RetryCallState) -> None:
        """Record sleep times for testing."""
        if retry_state.next_action and retry_state.next_action.sleep:
            self._sleep_times.append(retry_state.next_action.sleep)


def test_with_retries_helper_method():
    # Generator with no default policy
    generator = MockGenerator()
    new_generator = generator.with_retries(max_attempts=3)
    assert new_generator.retry_policy is not None
    assert new_generator.retry_policy.max_attempts == 3
    assert new_generator.retry_policy.base_delay == 1.0  # default

    # Generator with existing policy
    generator = MockGenerator(retry_policy=RetryPolicy(max_attempts=2, base_delay=0.5))
    new_generator = generator.with_retries(max_attempts=5)
    assert new_generator.retry_policy is not None
    assert new_generator.retry_policy.max_attempts == 5
    assert new_generator.retry_policy.base_delay == 0.5

    new_generator = generator.with_retries(4, base_delay=10)
    assert new_generator.retry_policy is not None
    assert new_generator.retry_policy.max_attempts == 4
    assert new_generator.retry_policy.base_delay == 10

    # Test max_delay parameter
    new_generator = generator.with_retries(3, max_delay=5.0)
    assert new_generator.retry_policy is not None
    assert new_generator.retry_policy.max_attempts == 3
    assert new_generator.retry_policy.base_delay == 0.5  # preserved
    assert new_generator.retry_policy.max_delay == 5.0


async def test_raises_exception_after_retries_exhausted():
    generator = MockGenerator(
        retry_policy=RetryPolicy(max_attempts=3, base_delay=1e-3),
    )
    generator._complete_mock.side_effect = RetriableError("Test error")

    with pytest.raises(RetriableError):
        await generator.complete(
            messages=[Message(role="user", content="Test message")]
        )

    assert generator._complete_mock.call_count == 3
    assert len(generator._sleep_times) == 2  # 2 sleeps for 3 attempts


async def test_raises_exception_if_not_retriable():
    generator = MockGenerator(
        retry_policy=RetryPolicy(max_attempts=3, base_delay=1e-3),
    )
    generator._complete_mock.side_effect = ValueError("Test error")

    with pytest.raises(ValueError):
        await generator.complete(
            messages=[Message(role="user", content="Test message")]
        )

    assert generator._complete_mock.call_count == 1


async def test_retries_with_result():
    generator = MockGenerator(
        retry_policy=RetryPolicy(max_attempts=3, base_delay=1e-3),
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
    assert len(generator._sleep_times) == 2  # 2 sleeps for 3 attempts


async def test_retries_works_with_batch_complete():
    generator = MockGenerator(
        retry_policy=RetryPolicy(max_attempts=3, base_delay=1e-3),
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
    assert len(generator._sleep_times) == 2  # 2 sleeps for 3 attempts


async def test_retries_with_max_delay():
    """Test that max_delay caps the exponential backoff."""
    generator = MockGenerator(
        retry_policy=RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=3.0),
    )
    generator._complete_mock.side_effect = [
        RetriableError("Test error"),
        RetriableError("Test error"),
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
    assert generator._complete_mock.call_count == 5

    # Verify that all sleep times are capped at max_delay
    for sleep_time in generator._sleep_times:
        assert sleep_time <= 3.0


async def test_retries_exponential_backoff():
    """Test that exponential backoff increases sleep times correctly."""
    generator = MockGenerator(
        retry_policy=RetryPolicy(max_attempts=4, base_delay=1.0),
    )
    generator._complete_mock.side_effect = [
        RetriableError("Test error"),
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
    assert generator._complete_mock.call_count == 4

    # Verify exponential growth: each sleep should be approximately double the previous
    assert len(generator._sleep_times) == 3
    for i in range(1, len(generator._sleep_times)):
        # Allow some tolerance for timing variations
        assert generator._sleep_times[i] >= generator._sleep_times[i - 1] * 1.5


async def test_retries_exponential_backoff_with_max_delay():
    """Test exponential backoff with max_delay capping."""
    generator = MockGenerator(
        retry_policy=RetryPolicy(max_attempts=6, base_delay=1.0, max_delay=5.0),
    )
    generator._complete_mock.side_effect = [
        RetriableError("Test error"),
        RetriableError("Test error"),
        RetriableError("Test error"),
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
    assert generator._complete_mock.call_count == 6

    # First sleeps should grow exponentially
    assert len(generator._sleep_times) == 5
    # Later sleeps should be capped at max_delay
    for sleep_time in generator._sleep_times[2:]:
        assert sleep_time <= 5.0
