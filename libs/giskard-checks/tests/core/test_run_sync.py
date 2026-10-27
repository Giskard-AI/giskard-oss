"""Tests for the synchronous ``run_sync`` entry points.

``Scenario``, ``Suite`` and ``TestCase`` expose ``run_sync`` as a convenience
wrapper over their async ``run``: it must behave like ``asyncio.run(obj.run())``
when no event loop is running, and refuse to run inside a running loop.
"""

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import pytest
from giskard.checks import (
    Equals,
    Interaction,
    Scenario,
    ScenarioResult,
    Suite,
    SuiteResult,
    TestCase,
    Trace,
)
from giskard.checks import (
    TestCaseResult as CheckTestCaseResult,
)
from pydantic.experimental.missing_sentinel import MISSING


def _scenario() -> Scenario[Any, Any, Any]:
    return (
        Scenario("run_sync_scenario")
        .interact("hello", lambda inputs: f"echo: {inputs}")
        .check(Equals(expected_value="echo: hello", target_key="trace.last.outputs"))
    )


def _suite() -> Suite[Any, Any]:
    return Suite(name="run_sync_suite").append(_scenario())


def _test_case() -> TestCase[Any, Any, Any]:
    trace = asyncio.run(
        Trace.from_interactions(Interaction(inputs="hello", outputs="echo: hello"))
    )
    return TestCase(
        name="run_sync_test_case",
        trace=trace,
        checks=[Equals(expected_value="echo: hello", target_key="trace.last.outputs")],
    )


def _passed(result: Any) -> bool:
    if isinstance(result, SuiteResult):
        return result.pass_rate == 1.0
    return result.passed


# (factory, result type, default kwargs `run_sync` forwards to `run`, instance name)
RUNNABLES: list[tuple[Callable[[], Any], type, dict[str, Any], str]] = [
    (
        _scenario,
        ScenarioResult,
        {"target": MISSING, "return_exception": False, "multiple_runs": None},
        "scenario",
    ),
    (
        _suite,
        SuiteResult,
        {
            "target": MISSING,
            "return_exception": False,
            "parallel": False,
            "max_concurrency": None,
            "verbose": True,
        },
        "suite",
    ),
    (_test_case, CheckTestCaseResult, {"return_exception": False}, "test_case"),
]
IDS = ["Scenario", "Suite", "TestCase"]


@pytest.mark.parametrize(
    ("factory", "result_type", "_defaults", "_name"), RUNNABLES, ids=IDS
)
def test_run_sync_returns_same_result_as_run(
    factory: Callable[[], Any], result_type: type, _defaults: Any, _name: str
) -> None:
    sync_result = factory().run_sync()
    async_result = asyncio.run(factory().run())

    assert isinstance(sync_result, result_type)
    assert _passed(sync_result)
    assert type(sync_result) is type(async_result)
    assert _passed(async_result)


@pytest.mark.parametrize(
    ("factory", "result_type", "_defaults", "_name"), RUNNABLES, ids=IDS
)
def test_run_sync_works_from_worker_thread(
    factory: Callable[[], Any], result_type: type, _defaults: Any, _name: str
) -> None:
    runnable = factory()
    outcome: dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["result"] = runnable.run_sync()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            outcome["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert isinstance(outcome["result"], result_type)
    assert _passed(outcome["result"])


@pytest.mark.parametrize(
    ("factory", "_result_type", "defaults", "_name"), RUNNABLES, ids=IDS
)
def test_run_sync_forwards_arguments_to_run(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[], Any],
    _result_type: type,
    defaults: dict[str, Any],
    _name: str,
) -> None:
    runnable = factory()
    calls: list[dict[str, Any]] = []
    sentinel = object()

    async def fake_run(self: Any, **kwargs: Any) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(type(runnable), "run", fake_run)

    assert runnable.run_sync() is sentinel
    target = lambda inputs: inputs  # noqa: E731
    overrides = {
        key: target
        if key == "target"
        else (not value if isinstance(value, bool) else 3)
        for key, value in defaults.items()
    }
    assert runnable.run_sync(**overrides) is sentinel

    assert calls == [defaults, {**defaults, **overrides}]


@pytest.mark.parametrize(
    ("factory", "_result_type", "_defaults", "_name"), RUNNABLES, ids=IDS
)
def test_run_sync_propagates_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[], Any],
    _result_type: type,
    _defaults: Any,
    _name: str,
) -> None:
    runnable = factory()

    async def failing_run(self: Any, **kwargs: Any) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(type(runnable), "run", failing_run)

    with pytest.raises(ValueError, match="boom"):
        runnable.run_sync()


@pytest.mark.parametrize(
    ("factory", "_result_type", "_defaults", "name"), RUNNABLES, ids=IDS
)
async def test_run_sync_refuses_running_event_loop(
    factory: Callable[[], Any], _result_type: type, _defaults: Any, name: str
) -> None:
    runnable = await _test_case_async() if name == "test_case" else factory()

    with pytest.raises(RuntimeError, match=rf"Use `await {name}\.run\(\.\.\.\)`"):
        runnable.run_sync()

    # The async API still works from the same loop.
    assert _passed(await runnable.run())


@pytest.mark.parametrize(
    ("factory", "_result_type", "_defaults", "name"), RUNNABLES, ids=IDS
)
async def test_run_sync_in_running_loop_does_not_create_coroutine(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[], Any],
    _result_type: type,
    _defaults: Any,
    name: str,
) -> None:
    runnable = await _test_case_async() if name == "test_case" else factory()
    calls: list[dict[str, Any]] = []

    async def fake_run(self: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(type(runnable), "run", fake_run)

    with pytest.raises(RuntimeError, match="event loop is running"):
        runnable.run_sync()

    # `run` is never called, so no coroutine is created and left un-awaited.
    assert calls == []


async def _test_case_async() -> TestCase[Any, Any, Any]:
    trace = await Trace.from_interactions(
        Interaction(inputs="hello", outputs="echo: hello")
    )
    return TestCase(
        name="run_sync_test_case",
        trace=trace,
        checks=[Equals(expected_value="echo: hello", target_key="trace.last.outputs")],
    )
