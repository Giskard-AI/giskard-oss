from unittest.mock import AsyncMock

import pytest
from giskard.agents.chat import Message
from giskard.agents.generators.base import BaseGenerator, GenerationParams, Response
from giskard.agents.generators.middleware import RetryMiddleware


class RetriableError(Exception):
    """A retriable error."""


class _RetriableOnlyMiddleware(RetryMiddleware):
    """Retry middleware that only retries RetriableError."""

    def _should_retry(self, err: Exception) -> bool:
        return isinstance(err, RetriableError)


class MockGenerator(BaseGenerator):
    """A mock generator for testing the retry middleware."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._complete_mock = AsyncMock()

    async def _complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        return await self._complete_mock(messages, params)


def _make_generator(**retry_kwargs) -> MockGenerator:
    mw = _RetriableOnlyMiddleware(**retry_kwargs) if retry_kwargs else _RetriableOnlyMiddleware()
    return MockGenerator(middleware=[mw])


async def test_raises_exception_after_retries_exhausted():
    generator = _make_generator(max_attempts=3, base_delay=1e-3)
    generator._complete_mock.side_effect = RetriableError("Test error")

    with pytest.raises(RetriableError):
        await generator.complete(
            messages=[Message(role="user", content="Test message")]
        )

    assert generator._complete_mock.call_count == 3


async def test_raises_exception_if_not_retriable():
    generator = _make_generator(max_attempts=3, base_delay=1e-3)
    generator._complete_mock.side_effect = ValueError("Test error")

    with pytest.raises(ValueError):
        await generator.complete(
            messages=[Message(role="user", content="Test message")]
        )

    assert generator._complete_mock.call_count == 1


async def test_retries_with_result():
    generator = _make_generator(max_attempts=3, base_delay=1e-3)
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
    generator = _make_generator(max_attempts=3, base_delay=1e-3)
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


async def test_retries_with_max_delay():
    """Test that max_delay caps the exponential backoff."""
    generator = _make_generator(max_attempts=5, base_delay=1e-3, max_delay=0.01)
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


async def test_retries_exponential_backoff():
    """Test that exponential backoff increases sleep times correctly."""
    generator = _make_generator(max_attempts=4, base_delay=1e-3)
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
