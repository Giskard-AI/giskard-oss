from typing import Any, override
from unittest.mock import AsyncMock, patch

import pytest
from giskard.agents.chat import Message
from giskard.agents.generators import FinishReason, GenerationParams
from giskard.agents.generators.base import BaseGenerator
from giskard.agents.generators.middleware import RetryMiddleware, RetryPolicy


class RetriableError(Exception):
    """A retriable error."""


class _RetriableOnlyMiddleware(RetryMiddleware):
    """Retry middleware that only retries RetriableError."""

    @override
    def _should_retry(self, err: Exception) -> bool:
        return isinstance(err, RetriableError)


class MockGenerator(BaseGenerator):
    """A mock generator for testing the retry middleware."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._call_model_mock = AsyncMock()

    @override
    async def _call_model(
        self,
        messages: list[Message],
        params: GenerationParams,
    ) -> tuple[Message, FinishReason]:
        return await self._call_model_mock(messages, params)


def _make_generator(**retry_kwargs: Any) -> MockGenerator:
    policy = RetryPolicy(**retry_kwargs) if retry_kwargs else RetryPolicy()
    mw = _RetriableOnlyMiddleware(retry_policy=policy)
    return MockGenerator(middlewares=[mw])


async def test_raises_exception_after_retries_exhausted():
    generator = _make_generator(max_attempts=3, base_delay=1e-3)
    generator._call_model_mock.side_effect = RetriableError("Test error")

    with pytest.raises(RetriableError):
        _ = await generator.complete(
            messages=[Message(role="user", content="Test message")]
        )

    assert generator._call_model_mock.call_count == 3


async def test_raises_exception_if_not_retriable():
    generator = _make_generator(max_attempts=3, base_delay=1e-3)
    generator._call_model_mock.side_effect = ValueError("Test error")

    with pytest.raises(ValueError):
        _ = await generator.complete(
            messages=[Message(role="user", content="Test message")]
        )

    assert generator._call_model_mock.call_count == 1


async def test_retries_with_result():
    generator = _make_generator(max_attempts=3, base_delay=1e-3)
    generator._call_model_mock.side_effect = [
        RetriableError("Test error"),
        RetriableError("Test error"),
        (Message(role="assistant", content="Test response"), "stop"),
    ]

    res = await generator.complete(
        messages=[Message(role="user", content="Test message")]
    )
    assert res.message.content == "Test response"
    assert res.finish_reason == "stop"

    assert generator._call_model_mock.call_count == 3


async def test_retries_works_with_batch_complete():
    generator = _make_generator(max_attempts=3, base_delay=1e-3)
    generator._call_model_mock.side_effect = [
        RetriableError("Test error"),
        RetriableError("Test error"),
        (Message(role="assistant", content="Test response"), "stop"),
    ]

    res = await generator.batch_complete(
        messages=[
            [Message(role="user", content="Test message")],
        ]
    )

    assert len(res) == 1
    assert res[0].message.content == "Test response"
    assert res[0].finish_reason == "stop"

    assert generator._call_model_mock.call_count == 3


async def test_retries_with_max_delay():
    """Test that max_delay caps the exponential backoff."""
    generator = _make_generator(max_attempts=5, base_delay=1.0, max_delay=3.0)
    generator._call_model_mock.side_effect = [
        RetriableError("Test error"),
        RetriableError("Test error"),
        RetriableError("Test error"),
        RetriableError("Test error"),
        (Message(role="assistant", content="Test response"), "stop"),
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        res = await generator.complete(
            messages=[Message(role="user", content="Test message")]
        )

    assert res.message.content == "Test response"
    assert generator._call_model_mock.call_count == 5

    for call in mock_sleep.call_args_list:
        assert call.args[0] <= 3.0


async def test_retries_exponential_backoff():
    """Test that exponential backoff increases sleep times correctly."""
    generator = _make_generator(max_attempts=4, base_delay=1.0)
    generator._call_model_mock.side_effect = [
        RetriableError("Test error"),
        RetriableError("Test error"),
        RetriableError("Test error"),
        (Message(role="assistant", content="Test response"), "stop"),
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        res = await generator.complete(
            messages=[Message(role="user", content="Test message")]
        )

    assert res.message.content == "Test response"
    assert generator._call_model_mock.call_count == 4

    sleep_times = [call.args[0] for call in mock_sleep.call_args_list]
    assert len(sleep_times) == 3  # 3 sleeps for 4 attempts
    for i in range(1, len(sleep_times)):
        assert sleep_times[i] >= sleep_times[i - 1] * 1.5


async def test_retries_exponential_backoff_with_max_delay():
    """Test exponential backoff with max_delay capping."""
    generator = _make_generator(max_attempts=6, base_delay=1.0, max_delay=5.0)
    generator._call_model_mock.side_effect = [
        RetriableError("Test error"),
        RetriableError("Test error"),
        RetriableError("Test error"),
        RetriableError("Test error"),
        RetriableError("Test error"),
        (Message(role="assistant", content="Test response"), "stop"),
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        res = await generator.complete(
            messages=[Message(role="user", content="Test message")]
        )

    assert res.message.content == "Test response"
    assert generator._call_model_mock.call_count == 6

    sleep_times = [call.args[0] for call in mock_sleep.call_args_list]
    assert len(sleep_times) == 5  # 5 sleeps for 6 attempts
    for sleep_time in sleep_times[2:]:
        assert sleep_time <= 5.0
