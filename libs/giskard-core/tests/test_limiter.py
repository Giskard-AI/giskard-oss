import asyncio
import sys
import time

import pytest
from giskard.core import RateLimiter

JITTER_TIME = 0.02  # 20ms jitter


class TestRateLimiter:
    def test_cannot_create_rate_limiter_with_duplicate_id(self):
        with pytest.raises(
            ValueError,
            match="Rate limiter with id 'test' already registered with different rules",
        ):
            _ = RateLimiter.min_interval(min_interval=1, id="test")
            _ = RateLimiter.min_interval(min_interval=2, id="test")

        with pytest.raises(
            ValueError,
            match="Rate limiter with id 'min_interval_1' already registered with different rules",
        ):
            _ = RateLimiter.min_interval(min_interval=1, id="min_interval_1")
            _ = RateLimiter.max_concurrent(max_concurrent=1, id="min_interval_1")

        with pytest.raises(
            ValueError,
            match="Rate limiter with id 'combined_1' already registered with different rules",
        ):
            _ = RateLimiter.from_rpm(rpm=1, max_concurrent=1, id="combined_1")
            _ = RateLimiter.from_rpm(rpm=1, id="combined_1")

    def test_same_rate_limiter_rules_with_same_id_should_not_raise_error(
        self,
    ):
        _ = RateLimiter.min_interval(min_interval=1, id="min_interval_1")
        _ = RateLimiter.min_interval(min_interval=1, id="min_interval_1")

        _ = RateLimiter.max_concurrent(max_concurrent=1, id="max_concurrent_1")
        _ = RateLimiter.max_concurrent(max_concurrent=1, id="max_concurrent_1")

        _ = RateLimiter.from_rpm(rpm=1, max_concurrent=1, id="combined_1")
        _ = RateLimiter.from_rpm(rpm=1, max_concurrent=1, id="combined_1")


class TestMinInterval:
    @pytest.mark.parametrize("rpm", [0, -1, -sys.maxsize / 2])
    def test_rpm_cannot_be_less_than_1(self, rpm: int):
        with pytest.raises(ValueError, match="RPM must be greater than 0"):
            _ = RateLimiter.from_rpm(rpm)

    @pytest.mark.timeout(1)
    async def test_allow_parrallel_requests(self):
        job_started_signal = asyncio.Event()
        signal = asyncio.Event()

        async def wait_for_signal(rate_limiter: RateLimiter) -> None:
            async with rate_limiter.throttle():
                # Ensure the job has started
                job_started_signal.set()
                _ = await signal.wait()

        async def signal_task(rate_limiter: RateLimiter) -> None:
            async with rate_limiter.throttle():
                signal.set()

        rate_limiter = RateLimiter.from_rpm(10_000)

        async with asyncio.TaskGroup() as tg:
            _ = tg.create_task(wait_for_signal(rate_limiter))
            _ = await job_started_signal.wait()
            _ = tg.create_task(signal_task(rate_limiter))

    async def test_throttle_rate(self):
        rate_limiter = RateLimiter.from_rpm(6_000)  # 10ms throttle rate

        waited_times: list[float] = []

        async def throttle_task(rate_limiter: RateLimiter) -> None:
            async with rate_limiter.throttle() as waited_time:
                assert len(waited_time) == 1
                assert waited_time[0][0].kind == "min_interval"
                waited_times.append(waited_time[0][1])

        start_time = time.monotonic()
        async with asyncio.TaskGroup() as tg:
            for _ in range(50):
                _ = tg.create_task(throttle_task(rate_limiter))

        elapsed_time = time.monotonic() - start_time
        assert (
            elapsed_time < 0.49 + JITTER_TIME
        )  # Ensure reasonable time to run the tasks

        assert len(waited_times) == 50
        assert waited_times[0] == 0  # Ensure the first request is not throttled
        for waited_time in waited_times[1:]:
            assert waited_time > 0
            assert (
                waited_time <= 0.49 + JITTER_TIME
            )  # Ensure the wait time is within the expected range

    async def test_throttle_rate_reset_after_interval(self):
        rate_limiter = RateLimiter.from_rpm(60 * 25)  # 25 requests per second

        async def throttle_task(rate_limiter: RateLimiter):
            async with rate_limiter.throttle() as waited_time:
                assert len(waited_time) == 1
                assert waited_time[0][0].kind == "min_interval"
                assert waited_time[0][1] == 0  # Ensure the task is not throttled

        for _ in range(10):
            await throttle_task(rate_limiter)
            await asyncio.sleep(
                1.0 / 25
            )  # Simulate delay between requests to ensure the next request is not throttled

    async def test_serialization_keeps_rate_limiter_instance(self):
        rate_limiter = RateLimiter.from_rpm(6_000)
        waited_times: list[float] = []

        async def throttle_task(rate_limiter: RateLimiter) -> None:
            deserialized_rate_limiter = RateLimiter.model_validate_json(
                rate_limiter.model_dump_json()
            )
            async with deserialized_rate_limiter.throttle() as waited_time:
                assert len(waited_time) == 1
                assert waited_time[0][0].kind == "min_interval"
                waited_times.append(waited_time[0][1])

        async with asyncio.TaskGroup() as tg:
            for _ in range(10):
                _ = tg.create_task(throttle_task(rate_limiter))

        assert len(waited_times) == 10
        assert waited_times[0] == 0  # Ensure the task is not throttled
        for waited_time in waited_times[1:]:
            assert waited_time > 0, waited_times  # Ensure the task is throttled


class TestMaxConcurrentRequests:
    @pytest.mark.parametrize("max_concurrent", [0, -1, -sys.maxsize / 2])
    def test_max_concurrent_cannot_be_less_than_1(self, max_concurrent: int):
        with pytest.raises(
            ValueError, match="Input should be greater than or equal to 1"
        ):
            _ = RateLimiter.max_concurrent(max_concurrent=max_concurrent)

    @pytest.mark.timeout(1)
    async def test_allow_parrallel_requests(self):
        barrier = asyncio.Barrier(10)

        async def throttle_task(rate_limiter: RateLimiter) -> None:
            async with rate_limiter.throttle() as waited_time:
                assert len(waited_time) == 1
                assert waited_time[0][0].kind == "max_concurrent"
                assert waited_time[0][1] == 0
                _ = await barrier.wait()

        rate_limiter = RateLimiter.max_concurrent(max_concurrent=10)

        async with asyncio.TaskGroup() as tg:
            for _ in range(10):
                _ = tg.create_task(throttle_task(rate_limiter))

    @pytest.mark.timeout(1)
    async def test_block_when_max_concurrent_reached(self):
        rate_limiter = RateLimiter.max_concurrent(max_concurrent=5)
        barrier = asyncio.Barrier(10)
        waited_times: list[float] = []

        async def throttle_task(rate_limiter: RateLimiter) -> None:
            async with rate_limiter.throttle() as waited_time:
                assert len(waited_time) == 1
                assert waited_time[0][0].kind == "max_concurrent"
                waited_times.append(waited_time[0][1])
                _ = await barrier.wait()

        async with asyncio.TaskGroup() as tg:
            for _ in range(10):
                _ = tg.create_task(throttle_task(rate_limiter))

            await asyncio.sleep(JITTER_TIME)
            assert barrier.n_waiting == 5  # Ensure only 5 tasks are running
            assert len(waited_times) == 5
            for waited_time in waited_times:
                assert waited_time == 0  # Ensure the tasks are not throttled
            waited_times.clear()

            for _ in range(5):
                _ = tg.create_task(barrier.wait())  # Unblock the tasks

            await asyncio.sleep(JITTER_TIME)
            assert barrier.n_waiting == 5  # Ensure the other 5 tasks are running
            assert len(waited_times) == 5
            for waited_time in waited_times:
                assert (
                    waited_time > JITTER_TIME and waited_time <= 2 * JITTER_TIME
                )  # Ensure the tasks are throttled

            for _ in range(5):
                _ = tg.create_task(barrier.wait())  # Unblock the tasks

    async def test_serialization_keeps_rate_limiter_instance(self):
        rate_limiter = RateLimiter.max_concurrent(max_concurrent=1)
        waited_times: list[float] = []
        barrier = asyncio.Barrier(2)

        async def throttle_task(rate_limiter: RateLimiter) -> None:
            deserialized_rate_limiter = RateLimiter.model_validate_json(
                rate_limiter.model_dump_json()
            )
            async with deserialized_rate_limiter.throttle() as waited_time:
                assert len(waited_time) == 1
                assert waited_time[0][0].kind == "max_concurrent"
                waited_times.append(waited_time[0][1])
                _ = await barrier.wait()

        async with asyncio.TaskGroup() as tg:
            for _ in range(2):
                _ = tg.create_task(throttle_task(rate_limiter))

            await asyncio.sleep(JITTER_TIME)
            assert barrier.n_waiting == 1
            assert len(waited_times) == 1
            assert waited_times[0] == 0
            waited_times.clear()

            _ = await barrier.wait()

            await asyncio.sleep(JITTER_TIME)
            assert barrier.n_waiting == 1
            assert len(waited_times) == 1
            assert waited_times[0] > JITTER_TIME and waited_times[0] <= 2 * JITTER_TIME
            waited_times.clear()

            _ = await barrier.wait()
