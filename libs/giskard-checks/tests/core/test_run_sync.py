from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock

import pytest
from giskard.checks import (
    Scenario,
    ScenarioResult,
    Suite,
    SuiteResult,
    TestCase,
    Trace,
)
from giskard.checks import TestCaseResult as CaseResult


def _make_scenario():
    return Scenario("sync scenario")


def _make_suite():
    return Suite(name="sync suite")


def _make_test_case():
    return TestCase(name="sync test case", trace=Trace(interactions=[]), checks=[])


def _echo_target(inputs):
    return inputs


@pytest.mark.parametrize(
    ("factory", "expected_type", "kwargs"),
    [
        pytest.param(_make_scenario, ScenarioResult, {}, id="scenario"),
        pytest.param(_make_suite, SuiteResult, {"verbose": False}, id="suite"),
        pytest.param(_make_test_case, CaseResult, {}, id="test-case"),
    ],
)
def test_run_sync_executes(factory, expected_type, kwargs):
    result = factory().run_sync(**kwargs)

    assert isinstance(result, expected_type)


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
                "max_concurrency": 2,
                "verbose": False,
            },
            id="suite",
        ),
        pytest.param(
            _make_test_case,
            (),
            {"return_exception": True},
            id="test-case",
        ),
    ],
)
def test_run_sync_forwards_arguments_and_result(monkeypatch, factory, args, kwargs):
    runnable = factory()
    expected = object()
    run = AsyncMock(return_value=expected)
    monkeypatch.setattr(type(runnable), "run", run)

    result = runnable.run_sync(*args, **kwargs)

    assert result is expected
    run.assert_awaited_once_with(*args, **kwargs)


@pytest.mark.parametrize(
    "factory",
    [_make_scenario, _make_suite, _make_test_case],
    ids=["scenario", "suite", "test-case"],
)
def test_run_sync_propagates_exception(monkeypatch, factory):
    runnable = factory()
    expected = ValueError("run failed")
    run = AsyncMock(side_effect=expected)
    monkeypatch.setattr(type(runnable), "run", run)

    with pytest.raises(ValueError) as exc_info:
        runnable.run_sync()

    assert exc_info.value is expected
    run.assert_awaited_once_with()


@pytest.mark.parametrize(
    ("factory", "expected_type", "kwargs"),
    [
        pytest.param(_make_scenario, ScenarioResult, {}, id="scenario"),
        pytest.param(_make_suite, SuiteResult, {"verbose": False}, id="suite"),
        pytest.param(_make_test_case, CaseResult, {}, id="test-case"),
    ],
)
def test_run_sync_works_in_non_main_thread(factory, expected_type, kwargs):
    runnable = factory()

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(runnable.run_sync, **kwargs).result(timeout=5)

    assert isinstance(result, expected_type)


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
