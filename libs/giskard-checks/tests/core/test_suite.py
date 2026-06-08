import asyncio
import time
from contextlib import nullcontext

import pytest
from giskard.checks import Equals, Scenario, Suite
from giskard.checks.core.result import ScenarioStatus
from giskard.checks.scenarios.suite import _OverallOnly, _SuiteProgress
from rich.progress import MofNCompleteColumn, Progress
from rich.text import Text


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

    async def delayed_identity(inputs):
        await asyncio.sleep(sleep_s)
        return inputs

    suite = Suite(name="parallel_speed_suite", target=delayed_identity)
    suite.append(Scenario("a").interact("a"))
    suite.append(Scenario("b").interact("b"))
    suite.append(Scenario("c").interact("c"))

    start = time.perf_counter()
    await suite.run(parallel=True)
    parallel_duration = time.perf_counter() - start

    # Must complete faster than running all scenarios serially
    assert parallel_duration < sleep_s * n


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


def test_progress_counter_appears_only_on_the_overall_row():
    """The counter shows on the overall summary row and is blank on scenario rows."""
    column = _OverallOnly(MofNCompleteColumn())
    with Progress(column, disable=True) as progress:
        overall_id = progress.add_task("overall", total=3, completed=1, overall=True)
        scenario_id = progress.add_task("scenario", total=None)
        by_id = {task.id: task for task in progress.tasks}

    overall = column.render(by_id[overall_id])
    scenario = column.render(by_id[scenario_id])
    assert isinstance(overall, Text)
    assert isinstance(scenario, Text)
    assert overall.plain == "1/3"
    assert scenario.plain == ""


@pytest.mark.asyncio
async def test_suite_progress_can_be_disabled(monkeypatch):
    """`verbose=False` builds a disabled bar so its calls are no-ops."""
    bars = []
    monkeypatch.setattr(Progress, "__enter__", lambda self: bars.append(self) or self)

    suite = Suite(name="progress_off_suite", target=lambda inputs: inputs)
    suite.append(Scenario("a").interact("a"))

    await suite.run(verbose=False)

    assert bars[0].disable is True


@pytest.mark.asyncio
async def test_suite_parallel_progress_shows_a_row_per_scenario(monkeypatch):
    """Parallel mode adds one progress row per running scenario."""
    rows = []
    original_add_task = Progress.add_task

    def record_row(self, description, **kwargs):
        rows.append(description)
        return original_add_task(self, description, **kwargs)

    monkeypatch.setattr(Progress, "add_task", record_row)

    suite = Suite(name="progress_rows_suite", target=lambda inputs: inputs)
    for name in ("alpha", "beta"):
        suite.append(Scenario(name).interact("hi"))

    await suite.run(parallel=True)

    assert "  ↳ alpha" in rows
    assert "  ↳ beta" in rows


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([], None),
        (
            [
                ScenarioStatus.FAIL,
                ScenarioStatus.PASS,
                ScenarioStatus.PASS,
                ScenarioStatus.PASS,
            ],
            "1 failed, 3 passed",
        ),
        (
            [ScenarioStatus.ERROR]
            + [ScenarioStatus.SKIP] * 2
            + [ScenarioStatus.FAIL] * 5
            + [ScenarioStatus.PASS] * 19,
            "1 errored, 5 failed, 2 skipped, 19 passed",
        ),
    ],
    ids=["empty", "omits_zero_counts", "all_nonzero_in_order"],
)
def test_progress_summary_renderable(records, expected):
    """Live summary lists errored, failed, skipped, passed; omits zeros; absent when empty."""
    progress = _SuiteProgress(disable=True)
    for status in records:
        progress.record(status)

    summary = progress._summary_renderable()
    if expected is None:
        assert summary is None
    else:
        assert summary is not None
        assert summary.plain.strip() == expected


@pytest.mark.asyncio
async def test_suite_progress_records_each_scenario_outcome(monkeypatch):
    """Each finished scenario feeds its status into the live counts, in order."""
    recorded = []
    monkeypatch.setattr(
        _SuiteProgress, "record", lambda self, status: recorded.append(status)
    )

    passing = Scenario("s1").interact("a", "a")
    failing = (
        Scenario("s2")
        .interact("b", "c")
        .check(Equals(expected_value="b", key="trace.last.outputs"))
    )
    suite = Suite(name="record_suite")
    suite.append(passing).append(failing)

    await suite.run()

    assert recorded == [ScenarioStatus.PASS, ScenarioStatus.FAIL]
