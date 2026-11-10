import asyncio
from collections.abc import Sequence
from typing import Any, override
from unittest.mock import AsyncMock, patch

import pytest
from giskard.agents.generators import GenerationParams
from giskard.agents.generators.base import BaseGenerator
from giskard.agents.generators.middleware import RetryMiddleware, RetryPolicy
from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    Choice,
    CompletionResponse,
    UserMessage,
)


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
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        return await self._call_model_mock(messages, params)


def _make_generator(**retry_kwargs: Any) -> MockGenerator:
    policy = RetryPolicy(**retry_kwargs) if retry_kwargs else RetryPolicy()
    mw = _RetriableOnlyMiddleware(retry_policy=policy)
    return MockGenerator(middlewares=[mw])


def _make_cancellable_generator() -> tuple[MockGenerator, asyncio.Event]:
    """Block the first model call, but let any unintended retry finish."""
    generator = MockGenerator(retry_policy=RetryPolicy(base_delay=0))
    entered = asyncio.Event()

    async def complete(
        messages: Sequence[ChatMessage], params: GenerationParams
    ) -> CompletionResponse:
        if generator._call_model_mock.await_count == 1:
            entered.set()
            await asyncio.Event().wait()
        return CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content="Unexpected retry"),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )

    generator._call_model_mock.side_effect = complete
    return generator, entered


async def test_cancelled_completion_is_not_retried():
    """A caller's cancellation must stop the active completion."""
    generator, entered = _make_cancellable_generator()
    task = asyncio.create_task(generator.complete([]))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert generator._call_model_mock.await_count == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_completion_deadline_is_not_swallowed():
    """Retrying must not turn an expired asyncio timeout into success."""
    generator, _ = _make_cancellable_generator()
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0):
            await generator.complete([])
    assert generator._call_model_mock.await_count == 1


class _AbortGeneration(BaseException):
    """Stand in for non-Exception control flow without interrupting pytest."""


@pytest.mark.parametrize("error", [asyncio.CancelledError(), _AbortGeneration()])
async def test_control_flow_does_not_reach_retry_predicate(error: BaseException):
    """Even a permissive custom retry predicate only receives Exceptions."""
    generator = MockGenerator(retry_policy=RetryPolicy(base_delay=0))
    generator._call_model_mock.side_effect = error
    with patch.object(RetryMiddleware, "_should_retry", return_value=True) as retry:
        with pytest.raises(type(error)) as exc_info:
            await generator.complete([])
        assert exc_info.value is error
        retry.assert_not_called()
    assert generator._call_model_mock.await_count == 1


async def test_default_policy_still_retries_ordinary_exceptions():
    """The generic middleware continues to retry regular model failures."""
    generator = MockGenerator(retry_policy=RetryPolicy(base_delay=0))
    response = CompletionResponse(choices=[])
    generator._call_model_mock.side_effect = [ValueError("Transient failure"), response]
    assert await generator.complete([]) is response
    assert generator._call_model_mock.await_count == 2


async def test_raises_exception_after_retries_exhausted():
    generator = _make_generator(max_attempts=3, base_delay=1e-3)
    generator._call_model_mock.side_effect = RetriableError("Test error")

    with pytest.raises(RetriableError):
        _ = await generator.complete(
            messages=[{"role": "user", "content": "Test message"}]
        )

    assert generator._call_model_mock.call_count == 3


async def test_raises_exception_if_not_retriable():
    generator = _make_generator(max_attempts=3, base_delay=1e-3)
    generator._call_model_mock.side_effect = ValueError("Test error")

    with pytest.raises(ValueError):
        _ = await generator.complete(
            messages=[{"role": "user", "content": "Test message"}]
        )

    assert generator._call_model_mock.call_count == 1


async def test_retries_with_result():
    generator = _make_generator(max_attempts=3, base_delay=1e-3)
    generator._call_model_mock.side_effect = [
        RetriableError("Test error"),
        RetriableError("Test error"),
        CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content="Test response"),
                    finish_reason="stop",
                    index=0,
                )
            ]
        ),
    ]

    res = await generator.complete(
        messages=[{"role": "user", "content": "Test message"}]
    )
    assert res.choices[0].message.content == "Test response"
    assert res.choices[0].finish_reason == "stop"

    assert generator._call_model_mock.call_count == 3


async def test_retries_works_with_batch_complete():
    generator = _make_generator(max_attempts=3, base_delay=1e-3)
    generator._call_model_mock.side_effect = [
        RetriableError("Test error"),
        RetriableError("Test error"),
        CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content="Test response"),
                    finish_reason="stop",
                    index=0,
                )
            ]
        ),
    ]

    res = await generator.batch_complete(
        messages=[
            [UserMessage(content="Test message")],
        ]
    )

    assert len(res) == 1
    assert res[0].choices[0].message.content == "Test response"
    assert res[0].choices[0].finish_reason == "stop"

    assert generator._call_model_mock.call_count == 3


async def test_retries_with_max_delay():
    """Test that max_delay caps the exponential backoff."""
    generator = _make_generator(max_attempts=5, base_delay=1.0, max_delay=3.0)
    generator._call_model_mock.side_effect = [
        RetriableError("Test error"),
        RetriableError("Test error"),
        RetriableError("Test error"),
        RetriableError("Test error"),
        CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content="Test response"),
                    finish_reason="stop",
                    index=0,
                )
            ]
        ),
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        res = await generator.complete(
            messages=[{"role": "user", "content": "Test message"}]
        )

    assert res.choices[0].message.content == "Test response"
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
        CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content="Test response"),
                    finish_reason="stop",
                    index=0,
                )
            ]
        ),
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        res = await generator.complete(
            messages=[{"role": "user", "content": "Test message"}]
        )

    assert res.choices[0].message.content == "Test response"
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
        CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content="Test response"),
                    finish_reason="stop",
                    index=0,
                )
            ]
        ),
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        res = await generator.complete(
            messages=[{"role": "user", "content": "Test message"}]
        )

    assert res.choices[0].message.content == "Test response"
    assert generator._call_model_mock.call_count == 6

    sleep_times = [call.args[0] for call in mock_sleep.call_args_list]
    assert len(sleep_times) == 5  # 5 sleeps for 6 attempts
    for sleep_time in sleep_times[2:]:
        assert sleep_time <= 5.0
