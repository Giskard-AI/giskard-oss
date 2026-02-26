import asyncio
import time
import uuid
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import override
from unittest.mock import patch

import pytest
from giskard.core import BaseRateLimiter, MinIntervalRateLimiter

JITTER_TIME = 0.02  # 20ms jitter


def _uid() -> str:
    """Generate a unique id per call to isolate tests from the singleton registry."""
    return str(uuid.uuid4())


@BaseRateLimiter.register("custom_rate_limiter")
class CustomRateLimiter(BaseRateLimiter):
    @override
    @asynccontextmanager
    async def throttle(self) -> AsyncGenerator[float]:
        yield 0.0


class TestRateLimiterRegistry:
    def test_warns_when_creating_rate_limiter_with_duplicate_id(self):
        rl_id = _uid()
        with pytest.warns(
            RuntimeWarning,
            match=f"Rate limiter with id '{rl_id}' already registered",
        ):
            _rate_limiter_a = MinIntervalRateLimiter.from_rpm(60, id=rl_id)
            _rate_limiter_b = MinIntervalRateLimiter.from_rpm(120, id=rl_id)

        rl_id = _uid()
        with pytest.warns(
            RuntimeWarning,
            match=f"Rate limiter with id '{rl_id}' already registered",
        ):
            _rate_limiter_a = MinIntervalRateLimiter.from_rpm(rpm=60, id=rl_id)
            _rate_limiter_b = MinIntervalRateLimiter.from_rpm(
                rpm=60, max_concurrent=1, id=rl_id
            )

        rl_id = _uid()
        with pytest.warns(
            RuntimeWarning,
            match=f"Rate limiter with id '{rl_id}' already registered",
        ):
            _rate_limiter_a = MinIntervalRateLimiter.from_rpm(rpm=60, id=rl_id)
            _rate_limiter_b = CustomRateLimiter(id=rl_id)

    def test_does_not_warn_when_disabled_and_creating_rate_limiter_with_duplicate_id(
        self,
    ):
        with patch(
            "giskard.core.rate_limiter.base.GISKARD_DISABLE_DUPLICATE_RATE_LIMITERS_WARNINGS",
            True,
        ):
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                rl_id = _uid()
                _rate_limiter_a = MinIntervalRateLimiter.from_rpm(60, id=rl_id)
                _rate_limiter_b = MinIntervalRateLimiter.from_rpm(120, id=rl_id)
            assert not any("already registered" in str(w.message) for w in record)

            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                rl_id = _uid()
                _rate_limiter_a = MinIntervalRateLimiter.from_rpm(rpm=60, id=rl_id)
                _rate_limiter_b = MinIntervalRateLimiter.from_rpm(
                    rpm=60, max_concurrent=1, id=rl_id
                )
            assert not any("already registered" in str(w.message) for w in record)

            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                rl_id = _uid()
                _rate_limiter_a = MinIntervalRateLimiter.from_rpm(rpm=60, id=rl_id)
                _rate_limiter_b = CustomRateLimiter(id=rl_id)
            assert not any("already registered" in str(w.message) for w in record)

    def test_same_rate_limiter_with_same_id_should_not_raise_error(
        self,
    ):
        rl_id = _uid()
        _custom_rate_limiter_a = CustomRateLimiter(id=rl_id)
        _custom_rate_limiter_b = CustomRateLimiter(id=rl_id)

        rl_id = _uid()
        _rate_limiter_a = MinIntervalRateLimiter.from_rpm(60, id=rl_id)
        _rate_limiter_b = MinIntervalRateLimiter.from_rpm(60, id=rl_id)

    def test_rate_limiter_should_cleanup_state_when_last_instance_is_deleted(self):
        rl_id = _uid()
        _rate_limiter_a = MinIntervalRateLimiter.from_rpm(
            60, max_concurrent=1, id=rl_id
        )
        assert _rate_limiter_a._state is not None
        assert _rate_limiter_a._state.semaphore is not None
        old_state = _rate_limiter_a._state
        del _rate_limiter_a

        _rate_limiter_b = MinIntervalRateLimiter.from_rpm(120, id=rl_id)
        assert _rate_limiter_b._state is not None
        assert _rate_limiter_b._state.semaphore is None  # Since max_concurrent is None
        assert _rate_limiter_b._state is not old_state

    def test_rate_limiter_should_share_state_between_instances(self):
        rl_id = _uid()
        _rate_limiter_a = MinIntervalRateLimiter.from_rpm(
            60, max_concurrent=1, id=rl_id
        )
        _rate_limiter_b = MinIntervalRateLimiter.from_rpm(
            60, max_concurrent=1, id=rl_id
        )
        assert _rate_limiter_a._state is _rate_limiter_b._state

    def test_rate_limiter_should_not_share_state_between_instances_with_different_ids(
        self,
    ):
        _rate_limiter_a = MinIntervalRateLimiter.from_rpm(
            60, max_concurrent=1, id=_uid()
        )
        _rate_limiter_b = MinIntervalRateLimiter.from_rpm(
            60, max_concurrent=1, id=_uid()
        )
        assert _rate_limiter_a._state is not _rate_limiter_b._state


class TestMinIntervalRateLimiter:
    """Tests for MinIntervalRateLimiter with min_interval (from_rpm), max_concurrent, and combined behavior."""

    @pytest.mark.parametrize("rpm", [0, -1])
    def test_rpm_must_be_positive(self, rpm: int):
        with pytest.raises(ValueError, match="RPM must be greater than 0"):
            _ = MinIntervalRateLimiter.from_rpm(rpm)

    @pytest.mark.parametrize("max_concurrent", [0, -1])
    def test_max_concurrent_cannot_be_less_than_1(self, max_concurrent: int):
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            _ = MinIntervalRateLimiter.from_rpm(60, max_concurrent=max_concurrent)

    @pytest.mark.timeout(1)
    async def test_min_interval_allows_parallel_requests(self):
        job_started_signal = asyncio.Event()
        signal = asyncio.Event()

        async def wait_for_signal(rate_limiter: MinIntervalRateLimiter) -> None:
            async with rate_limiter.throttle():
                job_started_signal.set()
                _ = await signal.wait()

        async def signal_task(rate_limiter: MinIntervalRateLimiter) -> None:
            async with rate_limiter.throttle():
                signal.set()

        rate_limiter = MinIntervalRateLimiter.from_rpm(10_000)

        async with asyncio.TaskGroup() as tg:
            _ = tg.create_task(wait_for_signal(rate_limiter))
            _ = await job_started_signal.wait()
            _ = tg.create_task(signal_task(rate_limiter))

    async def test_min_interval_throttles_rapid_requests(self):
        rate_limiter = MinIntervalRateLimiter.from_rpm(6_000)  # 10ms min interval

        all_waited: list[float] = []

        async def throttle_task(rate_limiter: MinIntervalRateLimiter) -> None:
            async with rate_limiter.throttle() as waited:
                all_waited.append(waited)

        start_time = time.monotonic()
        async with asyncio.TaskGroup() as tg:
            for _ in range(50):
                _ = tg.create_task(throttle_task(rate_limiter))

        elapsed_time = time.monotonic() - start_time
        assert elapsed_time < 0.49 + JITTER_TIME

        assert len(all_waited) == 50
        assert all_waited[0] < 1e-3  # First request not throttled
        for waited in all_waited[1:]:
            assert waited > 0
            assert waited <= 0.49 + JITTER_TIME

    async def test_min_interval_reset_after_interval(self):
        rate_limiter = MinIntervalRateLimiter.from_rpm(
            60 * 25
        )  # 25 requests per second

        async def throttle_task(rate_limiter: MinIntervalRateLimiter):
            async with rate_limiter.throttle() as waited:
                assert waited < 1e-3

        for _ in range(10):
            await throttle_task(rate_limiter)
            await asyncio.sleep(1.0 / 25)

    @pytest.mark.timeout(1)
    async def test_max_concurrent_allows_parallel_requests(self):
        barrier = asyncio.Barrier(10)

        async def throttle_task(rate_limiter: MinIntervalRateLimiter) -> None:
            async with rate_limiter.throttle() as waited:
                assert waited < 0.1  # May have small min_interval delay
                _ = await barrier.wait()

        rate_limiter = MinIntervalRateLimiter.from_rpm(10_000, max_concurrent=10)

        async with asyncio.TaskGroup() as tg:
            for _ in range(10):
                _ = tg.create_task(throttle_task(rate_limiter))

    @pytest.mark.timeout(1)
    async def test_max_concurrent_blocks_when_limit_reached(self):
        rate_limiter = MinIntervalRateLimiter.from_rpm(10_000, max_concurrent=5)
        barrier = asyncio.Barrier(10)
        all_waited: list[float] = []

        async def throttle_task(rate_limiter: MinIntervalRateLimiter) -> None:
            async with rate_limiter.throttle() as waited:
                all_waited.append(waited)
                _ = await barrier.wait()

        async with asyncio.TaskGroup() as tg:
            for _ in range(10):
                _ = tg.create_task(throttle_task(rate_limiter))

            await asyncio.sleep(0.08)  # Allow 5 tasks to enter (min_interval ~6ms each)
            assert barrier.n_waiting == 5
            assert len(all_waited) == 5
            for waited in all_waited:
                assert waited < 0.1  # First 5 have small min_interval delay
            all_waited.clear()

            for _ in range(5):
                _ = tg.create_task(barrier.wait())

            await asyncio.sleep(0.05)
            assert barrier.n_waiting == 5
            assert len(all_waited) == 5
            for waited in all_waited:
                assert waited > 0.05  # Throttled by semaphore + min_interval

            for _ in range(5):
                _ = tg.create_task(barrier.wait())

    async def test_combined_min_interval_and_max_concurrent(self):
        """Both limits apply: max 2 concurrent, min 20ms between starts."""
        rate_limiter = MinIntervalRateLimiter.from_rpm(
            50, max_concurrent=2
        )  # 20ms interval
        all_waited: list[float] = []

        async def throttle_task(rate_limiter: MinIntervalRateLimiter) -> None:
            async with rate_limiter.throttle() as waited:
                all_waited.append(waited)

        start = time.monotonic()
        async with asyncio.TaskGroup() as tg:
            for _ in range(4):
                _ = tg.create_task(throttle_task(rate_limiter))
        elapsed = time.monotonic() - start

        assert len(all_waited) == 4
        assert min(all_waited) < 0.05  # At least one (first) not throttled
        assert (
            max(all_waited) > 0.02
        )  # At least one throttled by semaphore + min_interval
        assert elapsed >= 0.03  # Min 2*20ms spacing for 4 tasks (2 bursts of 2)

    async def test_serialization_keeps_min_interval_behavior(self):
        rate_limiter = MinIntervalRateLimiter.from_rpm(6_000)
        all_waited: list[float] = []

        async def throttle_task(rate_limiter: MinIntervalRateLimiter) -> None:
            deserialized = MinIntervalRateLimiter.model_validate_json(
                rate_limiter.model_dump_json()
            )
            async with deserialized.throttle() as waited:
                all_waited.append(waited)

        async with asyncio.TaskGroup() as tg:
            for _ in range(10):
                _ = tg.create_task(throttle_task(rate_limiter))

        assert len(all_waited) == 10
        assert all_waited[0] < 1e-3
        for waited in all_waited[1:]:
            assert waited > 0

    async def test_serialization_keeps_max_concurrent_behavior(self):
        rate_limiter = MinIntervalRateLimiter.from_rpm(10_000, max_concurrent=1)
        all_waited: list[float] = []
        barrier = asyncio.Barrier(2)

        async def throttle_task(rate_limiter: MinIntervalRateLimiter) -> None:
            deserialized = MinIntervalRateLimiter.model_validate_json(
                rate_limiter.model_dump_json()
            )
            async with deserialized.throttle() as waited:
                all_waited.append(waited)
                _ = await barrier.wait()

        async with asyncio.TaskGroup() as tg:
            for _ in range(2):
                _ = tg.create_task(throttle_task(rate_limiter))

            await asyncio.sleep(JITTER_TIME)
            assert barrier.n_waiting == 1
            assert len(all_waited) == 1
            assert all_waited[0] <= 1e-3  # negligible wait
            all_waited.clear()

            _ = await barrier.wait()

            await asyncio.sleep(JITTER_TIME)
            assert len(all_waited) == 1
            assert all_waited[0] > 0.01  # Throttled by semaphore + min_interval

            _ = await barrier.wait()
