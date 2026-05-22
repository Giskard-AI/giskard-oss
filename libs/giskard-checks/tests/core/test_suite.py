import asyncio
import io
import sys
from contextlib import nullcontext

import pytest
from giskard.checks import Equals, Scenario, Suite


class RecordingStdout(io.StringIO):
    def __init__(self, *, is_tty: bool):
        super().__init__()
        self._is_tty = is_tty

    def isatty(self):
        return self._is_tty


@pytest.fixture
def sut1():
    return lambda inputs: f"SUT1: {inputs}"


@pytest.fixture
def sut2():
    return lambda inputs: f"SUT2: {inputs}"


@pytest.fixture
def sut3():
    return lambda inputs: f"SUT3: {inputs}"


@pytest.fixture
def identity_sut():
    return lambda inputs: inputs


@pytest.mark.asyncio
async def test_suite_target_precedence(sut1, sut2):
    """Verify that suite target overrides scenario target."""
    # Scenario with its own target passed directly to Scenario()
    scenario = (
        Scenario("test", target=sut1)
        .interact("hello")
        .check(Equals(expected_value="SUT2: hello", key="trace.last.outputs"))
    )

    # Suite with a different target
    suite = Suite(name="my_suite", target=sut2)
    suite.append(scenario)

    result = await suite.run()
    assert result.passed_count == 1
    assert result.results[0].passed


@pytest.mark.asyncio
async def test_suite_run_target_precedence(sut1, sut2, sut3):
    """Verify that run target overrides suite target."""
    scenario = (
        Scenario("test", target=sut1)
        .interact("hello")
        .check(Equals(expected_value="SUT3: hello", key="trace.last.outputs"))
    )

    suite = Suite(name="my_suite", target=sut2)
    suite.append(scenario)

    # Pass target to run()
    result = await suite.run(target=sut3)
    assert result.passed_count == 1
    assert result.results[0].passed


@pytest.mark.asyncio
async def test_suite_mixed_targets(sut1, sut2):
    """Verify that scenarios without suite-level target still work with their own targets."""
    scenario1 = (
        Scenario("s1", target=sut1)
        .interact("hello")
        .check(Equals(expected_value="SUT1: hello", key="trace.last.outputs"))
    )

    scenario2 = (
        Scenario("s2", target=sut2)
        .interact("world")
        .check(Equals(expected_value="SUT2: world", key="trace.last.outputs"))
    )

    # Suite with NO target
    suite = Suite(name="mixed_suite")
    suite.append(scenario1)
    suite.append(scenario2)

    result = await suite.run()
    assert result.passed_count == 2
    assert result.results[0].scenario_name == "s1"
    assert result.results[1].scenario_name == "s2"


@pytest.mark.asyncio
async def test_suite_result_aggregation():
    """Verify SuiteResult aggregation logic."""
    scenario1 = Scenario("s1").interact("a", "a")
    scenario2 = (
        Scenario("s2")
        .interact("b", "c")
        .check(Equals(expected_value="b", key="trace.last.outputs"))
    )

    suite = Suite(name="agg_suite")
    suite.append(scenario1)
    suite.append(scenario2)

    result = await suite.run()
    assert len(result.results) == 2
    assert result.skipped_count == 0
    assert result.passed_count == 1
    assert result.failed_count == 1
    assert result.pass_rate == 0.5
    assert result.results[0].scenario_name == "s1"
    assert result.results[1].scenario_name == "s2"
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_suite_callable_target():
    """Verify that suite target can be a callable."""
    scenario = Scenario("s1").interact("hello")

    # Suite with a callable target
    suite = Suite(name="callable_suite", target=lambda inputs: f"Callable: {inputs}")
    suite.append(scenario)

    result = await suite.run()
    assert result.passed_count == 1
    last_interaction = result.results[0].final_trace.last
    assert last_interaction is not None
    assert last_interaction.outputs == "Callable: hello"


def test_suite_append_returns_self():
    """Verify that append() returns the suite itself for fluent chaining."""
    suite = Suite(name="chain_suite")
    scenario_a = Scenario("a").interact("hello")

    result = suite.append(scenario_a)
    assert result is suite


@pytest.mark.asyncio
async def test_suite_append_chaining():
    """Verify that chained append() calls add all scenarios correctly."""
    scenario_a = Scenario("a", target=lambda inputs: inputs).interact("hello")
    scenario_b = Scenario("b", target=lambda inputs: inputs).interact("world")

    suite = Suite(name="chain_suite").append(scenario_a).append(scenario_b)

    assert len(suite.scenarios) == 2
    assert suite.scenarios[0] is scenario_a
    assert suite.scenarios[1] is scenario_b
    result = await suite.run()
    assert len(result.results) == 2
    assert result.results[0].scenario_name == "a"
    assert result.results[1].scenario_name == "b"


@pytest.mark.asyncio
async def test_suite_parallel_preserves_result_order():
    delays = {"first": 0.09, "second": 0.01, "third": 0.05}

    async def delayed_identity(inputs):
        await asyncio.sleep(delays[inputs])
        return inputs

    suite = Suite(name="parallel_order_suite", target=delayed_identity)
    suite.append(Scenario("first").interact("first"))
    suite.append(Scenario("second").interact("second"))
    suite.append(Scenario("third").interact("third"))

    result = await suite.run(parallel=True)

    assert [scenario.scenario_name for scenario in result.results] == [
        "first",
        "second",
        "third",
    ]
    assert result.passed_count == 3


@pytest.mark.asyncio
async def test_suite_parallel_runs_concurrently():
    sleep_s = 0.06
    n = 3
    active_runs = 0
    peak_runs = 0

    async def delayed_identity(inputs):
        nonlocal active_runs, peak_runs
        active_runs += 1
        peak_runs = max(peak_runs, active_runs)
        try:
            await asyncio.sleep(sleep_s)
            return inputs
        finally:
            active_runs -= 1

    suite = Suite(name="parallel_speed_suite", target=delayed_identity)
    suite.append(Scenario("a").interact("a"))
    suite.append(Scenario("b").interact("b"))
    suite.append(Scenario("c").interact("c"))

    await suite.run(parallel=True)

    assert peak_runs == n


@pytest.mark.asyncio
async def test_suite_parallel_fail_fast_when_return_exception_is_false():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def flaky_target(inputs):
        if inputs == "boom":
            raise RuntimeError("boom")
        started.set()
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return inputs

    suite = Suite(name="parallel_fail_fast_suite", target=flaky_target)
    suite.append(Scenario("slow").interact("slow"))
    suite.append(Scenario("boom").interact("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await suite.run(parallel=True)

    assert started.is_set()
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_suite_parallel_telemetry_includes_flag(monkeypatch):
    events = []

    def capture(event, *, properties):
        events.append((event, properties))

    monkeypatch.setattr(
        "giskard.checks.scenarios.suite.telemetry_capture",
        capture,
    )
    monkeypatch.setattr(
        "giskard.checks.scenarios.suite.telemetry_run_context",
        nullcontext,
    )
    monkeypatch.setattr(
        "giskard.checks.scenarios.suite.telemetry_tag",
        lambda *args, **kwargs: None,
    )

    suite = Suite(name="telemetry_suite", target=lambda inputs: inputs)
    suite.append(Scenario("a").interact("hello"))

    await suite.run(parallel=True)

    assert events[0][0] == "checks_suite_run_started"
    assert events[0][1]["parallel"] is True
    assert events[1][0] == "checks_suite_run_finished"
    assert events[1][1]["parallel"] is True


@pytest.mark.asyncio
async def test_suite_parallel_respects_max_concurrency():
    active_runs = 0
    peak_runs = 0

    async def tracked_target(inputs):
        nonlocal active_runs, peak_runs
        active_runs += 1
        peak_runs = max(peak_runs, active_runs)
        try:
            await asyncio.sleep(0.05)
            return inputs
        finally:
            active_runs -= 1

    suite = Suite(name="parallel_limit_suite", target=tracked_target)
    suite.append(Scenario("a").interact("a"))
    suite.append(Scenario("b").interact("b"))
    suite.append(Scenario("c").interact("c"))

    result = await suite.run(parallel=True, max_concurrency=2)

    assert result.passed_count == 3
    assert peak_runs == 2


@pytest.mark.asyncio
async def test_suite_parallel_rejects_invalid_max_concurrency():
    suite = Suite(name="invalid_parallel_limit_suite", target=lambda inputs: inputs)
    suite.append(Scenario("a").interact("a"))

    with pytest.raises(ValueError, match="max_concurrency must be greater than 0"):
        await suite.run(parallel=True, max_concurrency=0)


@pytest.mark.asyncio
async def test_suite_run_reports_live_progress_when_stdout_is_tty(monkeypatch):
    stdout = RecordingStdout(is_tty=True)
    monkeypatch.setattr(sys, "stdout", stdout)

    suite = Suite(name="progress_suite")
    suite.append(Scenario("pass").interact("pass", "pass"))
    suite.append(
        Scenario("fail")
        .interact("fail", "actual")
        .check(Equals(expected_value="expected", key="trace.last.outputs"))
    )

    await suite.run()

    output = stdout.getvalue()
    assert 'Running suite "progress_suite"' in output
    assert "2/2" in output
    assert "." in output
    assert "F" in output


@pytest.mark.asyncio
async def test_suite_run_progress_handles_braces_in_suite_name(monkeypatch):
    stdout = RecordingStdout(is_tty=True)
    monkeypatch.setattr(sys, "stdout", stdout)

    suite = Suite(name="progress {suite}")
    suite.append(Scenario("pass").interact("pass", "pass"))

    result = await suite.run()

    assert result.passed_count == 1
    assert 'Running suite "progress {suite}"' in stdout.getvalue()


@pytest.mark.asyncio
async def test_suite_run_suppresses_live_progress_when_not_tty(monkeypatch):
    stdout = RecordingStdout(is_tty=False)
    monkeypatch.setattr(sys, "stdout", stdout)

    suite = Suite(name="quiet_suite")
    suite.append(Scenario("pass").interact("pass", "pass"))

    await suite.run()

    assert stdout.getvalue() == ""


@pytest.mark.asyncio
async def test_suite_run_suppresses_live_progress_when_verbose_false(monkeypatch):
    stdout = RecordingStdout(is_tty=True)
    monkeypatch.setattr(sys, "stdout", stdout)

    suite = Suite(name="quiet_suite")
    suite.append(Scenario("pass").interact("pass", "pass"))

    await suite.run(verbose=False)

    assert stdout.getvalue() == ""


@pytest.mark.asyncio
async def test_suite_run_writes_status_symbol_before_next_serial_scenario(
    monkeypatch,
):
    stdout = RecordingStdout(is_tty=True)
    monkeypatch.setattr(sys, "stdout", stdout)
    output_before_second_start = None

    def target(inputs):
        nonlocal output_before_second_start
        if inputs == "second":
            output_before_second_start = stdout.getvalue()
        return inputs

    suite = Suite(name="incremental_suite", target=target)
    suite.append(Scenario("first").interact("first"))
    suite.append(Scenario("second").interact("second"))

    await suite.run()

    assert output_before_second_start is not None
    assert "." in output_before_second_start
