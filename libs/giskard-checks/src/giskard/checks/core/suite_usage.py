"""Suite-level LLM token usage aggregation."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, override

from giskard.agents import BaseGenerator
from giskard.llm.types import ChatMessage, CompletionResponse, Usage
from pydantic import BaseModel, ConfigDict, Field


class SuiteUsage(BaseModel, frozen=True):
    """Aggregated LLM token usage for a suite run."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @classmethod
    def from_llm_usage(cls, usage: Usage | None) -> "SuiteUsage":
        """Build a usage snapshot from a single ``giskard.llm`` ``Usage`` record."""
        if usage is None:
            return cls()
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens
        total_tokens = usage.total_tokens or (prompt_tokens + completion_tokens)
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def __add__(self, other: "SuiteUsage") -> "SuiteUsage":
        return SuiteUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class _SuiteUsageCollector:
    """Mutable accumulator for the active suite run."""

    def __init__(self) -> None:
        self._usage: SuiteUsage = SuiteUsage()

    def record(self, usage: Usage | None) -> None:
        if usage is None:
            return
        self._usage = self._usage + SuiteUsage.from_llm_usage(usage)

    def snapshot(self) -> SuiteUsage:
        return self._usage


_active_collector: ContextVar[_SuiteUsageCollector | None] = ContextVar(
    "_active_collector", default=None
)


def get_active_suite_usage_collector() -> _SuiteUsageCollector | None:
    """Return the collector for the current suite run, if any."""
    return _active_collector.get()


def record_completion_usage(usage: Usage | None) -> None:
    """Add one completion's usage to the active suite collector."""
    collector = _active_collector.get()
    if collector is not None:
        collector.record(usage)


@contextmanager
def suite_usage_collector() -> Iterator[_SuiteUsageCollector]:
    """Activate suite usage collection for the current async context."""
    collector = _SuiteUsageCollector()
    token = _active_collector.set(collector)
    try:
        yield collector
    finally:
        _active_collector.reset(token)


class _UsageRecordingGenerator(BaseGenerator):
    """Wraps a generator and forwards completion usage to the suite collector."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    wrapped: BaseGenerator

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: Any,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        response = await self.wrapped.complete(messages, params, metadata)
        record_completion_usage(response.usage)
        return response


def with_usage_recording(generator: BaseGenerator) -> BaseGenerator:
    """Return ``generator`` wrapped to record completion usage when a collector is active."""
    if isinstance(generator, _UsageRecordingGenerator):
        return generator
    return _UsageRecordingGenerator(wrapped=generator)
