import asyncio
import datetime
import time

from giskard.agents.rate_limiter import RateLimiter


class MockRateLimitError(Exception):
    pass


async def mock_job(rate_limiter: RateLimiter):
    async with rate_limiter.throttle():
        return datetime.datetime.now()


async def mock__long_job(rate_limiter: RateLimiter, wait_time: float):
    async with rate_limiter.throttle():
        started = time.monotonic()
        await asyncio.sleep(wait_time)
        return started


async def test_rate_limiter_max_concurrent_requests():
    rate_limiter = RateLimiter.from_rpm(500, max_concurrent=10)

    # Lock all threads
    for _ in range(10):
        await rate_limiter.acquire()

    # Create a task
    task = asyncio.create_task(mock_job(rate_limiter))

    # Task should be blocked
    assert not task.done()

    # Unlock a thread
    unlock_time = datetime.datetime.now()
    rate_limiter.release()

    # Task should be released
    await asyncio.wait_for(task, timeout=1.0)
    assert task.done()
    assert task.result() > unlock_time


async def test_rate_limiter_throttle_rate():
    rpm = 240
    expected_interval = 60.0 / rpm
    rate_limiter = RateLimiter.from_rpm(rpm, max_concurrent=2)

    start_time = time.monotonic()
    for i in range(10):
        async with rate_limiter.throttle():
            pass
        assert (
            time.monotonic() - start_time >= expected_interval * i
            and time.monotonic() - start_time < expected_interval * (i + 1)
        )

    # No throttle should be applied
    await asyncio.sleep(expected_interval)
    start_time = time.monotonic()
    async with rate_limiter.throttle():
        pass
    assert time.monotonic() - start_time < expected_interval


async def test_rate_limiter_requests_per_minute():
    """Test that rpm actually limits to requests per minute, not requests per second."""
    # Set a low rpm to make the timing difference obvious
    rpm = 60
    rate_limiter = RateLimiter.from_rpm(rpm, max_concurrent=10)

    start_time = time.monotonic()

    # Make 4 requests - with correct implementation this should take ~3 seconds
    # (3 intervals of 1 second each between 4 requests)
    for _ in range(4):
        async with rate_limiter.throttle():
            pass

    elapsed_time = time.monotonic() - start_time

    # With correct implementation: should take ~3 seconds (60/60 = 1 second per interval)
    # With buggy implementation: takes ~0.05 seconds (1/60 ≈ 0.0167 seconds per interval)
    assert elapsed_time >= 2.5, (
        f"Expected at least 2.5 seconds for 4 requests at 60 rpm, got {elapsed_time:.3f} seconds"
    )


async def test_rate_limiter_max_concurrent():
    rpm = 600
    expected_interval = 0.1
    rate_limiter = RateLimiter.from_rpm(rpm, max_concurrent=10)

    tasks = [
        asyncio.create_task(mock__long_job(rate_limiter, wait_time=3.0))
        for _ in range(20)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # First 10 requests should be run immediately every 0.1 seconds
    first_start = results[0]
    for i in range(10):
        assert results[i] - first_start >= expected_interval * i
        assert results[i] - first_start < expected_interval * (i + 1)

    # Next requests should wait first 10 tasks to finish, i.e. the 11th request
    # will start 5 seconds after the 1st request.
    second_batch_start = results[10]
    assert second_batch_start - first_start >= 3.0
    assert second_batch_start - first_start < 3.0 + expected_interval

    # Then the rest of the requests should be spaced by the expected interval
    for i in range(11, 20):
        assert results[i] - second_batch_start >= expected_interval * (i - 10)
        assert results[i] - second_batch_start < expected_interval * (i - 9)


def test_can_serialize_rate_limiter():
    rate_limiter = RateLimiter.from_rpm(
        rpm=600, max_concurrent=10, rate_limiter_id="test_can_serialize_rate_limiter"
    )
    assert rate_limiter.model_dump() == {
        "rate_limiter_id": "test_can_serialize_rate_limiter",
        "strategy": {"min_interval": 0.1, "max_concurrent": 10},
    }
