import pytest
from giskard.checks import Equals, Scenario, Suite
from giskard.checks.core.interaction import Trace
from giskard.checks.core.result import (
    CheckResult,
    ScenarioResult,
    SuiteResult,
)
from giskard.checks.core.result import (
    TestCaseResult as CheckTestCaseResult,
)
from rich.console import Console


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
async def test_suite_run_propagates_max_reported_failures():
    scenario = (
        Scenario("s1")
        .interact("b", "c")
        .check(Equals(expected_value="b", key="trace.last.outputs"))
    )

    suite = Suite(name="agg_suite", max_reported_failures=3)
    suite.append(scenario)

    result = await suite.run()

    assert result.max_reported_failures == 3


def test_suite_result_rich_console_respects_max_reported_failures():
    def failed_scenario(name: str) -> ScenarioResult[Trace]:
        return ScenarioResult(
            scenario_name=name,
            steps=[
                CheckTestCaseResult(
                    results=[
                        CheckResult.failure(
                            message=f"{name} failed",
                            details={"check_name": "ExampleCheck"},
                        )
                    ],
                    duration_ms=1,
                )
            ],
            duration_ms=1,
            final_trace=Trace(interactions=[]),
        )

    result = SuiteResult(
        results=[failed_scenario("s1"), failed_scenario("s2"), failed_scenario("s3")],
        duration_ms=3,
        max_reported_failures=2,
    )
    console = Console(record=True, width=120)

    console.print(result)

    output = console.export_text()
    assert "s1" in output
    assert "s2" in output
    assert "s3" not in output
    assert "... and 1 more" in output


def test_suite_result_rich_console_shows_all_failures_when_unbounded():
    def failed_scenario(name: str) -> ScenarioResult[Trace]:
        return ScenarioResult(
            scenario_name=name,
            steps=[
                CheckTestCaseResult(
                    results=[
                        CheckResult.failure(
                            message=f"{name} failed",
                            details={"check_name": "ExampleCheck"},
                        )
                    ],
                    duration_ms=1,
                )
            ],
            duration_ms=1,
            final_trace=Trace(interactions=[]),
        )

    result = SuiteResult(
        results=[failed_scenario("s1"), failed_scenario("s2"), failed_scenario("s3")],
        duration_ms=3,
        max_reported_failures=None,
    )
    console = Console(record=True, width=120)

    console.print(result)

    output = console.export_text()
    assert "s1" in output
    assert "s2" in output
    assert "s3" in output
    assert "... and" not in output


def test_suite_result_rich_console_can_hide_all_failure_details():
    def failed_scenario(name: str) -> ScenarioResult[Trace]:
        return ScenarioResult(
            scenario_name=name,
            steps=[
                CheckTestCaseResult(
                    results=[
                        CheckResult.failure(
                            message=f"{name} failed",
                            details={"check_name": "ExampleCheck"},
                        )
                    ],
                    duration_ms=1,
                )
            ],
            duration_ms=1,
            final_trace=Trace(interactions=[]),
        )

    result = SuiteResult(
        results=[failed_scenario("s1"), failed_scenario("s2"), failed_scenario("s3")],
        duration_ms=3,
        max_reported_failures=0,
    )
    console = Console(record=True, width=120)

    console.print(result)

    output = console.export_text()
    assert "s1" not in output
    assert "s2" not in output
    assert "s3" not in output
    assert "... and 3 more" in output
