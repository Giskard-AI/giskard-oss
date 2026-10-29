from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock

import pytest
from giskard.checks import Equals, Interaction, Scenario, Suite, TestCase, Trace
from giskard.checks.core._run_sync import run_sync


def _echo_target(inputs):
    return inputs


def _make_scenario(expected_value="hello"):
    return (
        Scenario("sync scenario")
        .interact("hello")
        .check(Equals(expected_value=expected_value, target_key="trace.last.outputs"))
    )


def _make_suite(expected_value="hello"):
    return Suite(name="sync suite").append(_make_scenario(expected_value))


def _make_test_case(expected_value="hello"):
    return TestCase(
        name="sync test case",
        trace=Trace(interactions=[Interaction(inputs="input", outputs="hello")]),
        checks=[Equals(expected_value=expected_value, target_key="trace.last.outputs")],
    )


@pytest.mark.parametrize("passed", [True, False], ids=["passing", "failing"])
@pytest.mark.parametrize("worker_thread", [False, True], ids=["main", "worker"])
@pytest.mark.parametrize(
    ("factory", "args", "kwargs"),
    [
        pytest.param(
            _make_scenario,
            (_echo_target,),
            {"return_exception": True, "multiple_runs": 2},
            id="scenario",
        ),
        pytest.param(
            _make_suite,
            (_echo_target,),
            {
                "return_exception": True,
                "parallel": True,
                "max_concurrency": 1,
                "verbose": False,
            },
            id="suite",
        ),
        pytest.param(_make_test_case, (True,), {}, id="test-case"),
    ],
)
def test_run_sync_executes_real_checks(factory, args, kwargs, worker_thread, passed):
    runnable = factory("hello" if passed else "different")

    if worker_thread:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(runnable.run_sync, *args, **kwargs).result(
                timeout=5
            )
    else:
        result = runnable.run_sync(*args, **kwargs)

    if isinstance(runnable, Suite):
        assert len(result.results) == 1
        assert result.passed_count == int(passed)
        assert result.pass_rate == int(passed)
        scenario_result = result.results[0]
    elif isinstance(runnable, Scenario):
        scenario_result = result
        assert result.multiple_runs == 2
        assert result.runs_executed == (2 if passed else 1)
    else:
        assert len(result.results) == 1
        assert result.passed is passed
        assert result.results[0].passed is passed
        assert result.results[0].failed is not passed
        return

    assert len(scenario_result.final_trace.interactions) == 1
    assert len(scenario_result.steps) == 1
    assert scenario_result.passed is passed
    check_results = scenario_result.steps[0].results
    assert len(check_results) == 1
    assert check_results[0].passed is passed
    assert check_results[0].failed is not passed


def test_shared_run_sync_forwards_arguments_and_result():
    expected = object()
    run = AsyncMock(return_value=expected)

    result = run_sync(run, "target", True, multiple_runs=2)

    assert result is expected
    run.assert_awaited_once_with("target", True, multiple_runs=2)


def test_shared_run_sync_propagates_exception():
    expected = ValueError("run failed")
    run = AsyncMock(side_effect=expected)

    with pytest.raises(ValueError) as exc_info:
        run_sync(run)

    assert exc_info.value is expected
    run.assert_awaited_once_with()


@pytest.mark.parametrize(
    "factory",
    [_make_scenario, _make_suite, _make_test_case],
    ids=["scenario", "suite", "test-case"],
)
async def test_run_sync_rejects_active_event_loop(monkeypatch, factory):
    runnable = factory()
    run = AsyncMock()
    monkeypatch.setattr(type(runnable), "run", run)

    with pytest.raises(
        RuntimeError,
        match=(
            r"^run_sync\(\) cannot be called while an asyncio event loop is running; "
            r"use await obj\.run\(\.\.\.\) instead\.$"
        ),
    ):
        runnable.run_sync()

    run.assert_not_called()
