import pytest
from giskard.checks import Equals, Scenario, Suite


@pytest.fixture
def sut1():
    return lambda x: f"SUT1: {x}"


@pytest.fixture
def sut2():
    return lambda x: f"SUT2: {x}"


@pytest.fixture
def sut3():
    return lambda x: f"SUT3: {x}"


@pytest.fixture
def identity_sut():
    return lambda x: x


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
    suite = Suite(name="callable_suite", target=lambda x: f"Callable: {x}")
    suite.append(scenario)

    result = await suite.run()
    assert result.passed_count == 1
    last_interaction = result.results[0].final_trace.last
    assert last_interaction is not None
    assert last_interaction.outputs == "Callable: hello"
