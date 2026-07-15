"""Opt-in progress logging in the scenario runner (issue #2421).

The runner emits INFO-level progress via the standard ``logging`` hierarchy,
so callers can surface per-step / per-check progress in CI, scripts, and
notebooks without the rich progress bar. It stays silent by default.
"""

import logging
from typing import Any

import pytest
from giskard.checks import Equals, Scenario, Suite


def _identity(inputs: Any) -> Any:
    return inputs


def _demo_suite() -> Suite[Any, Any]:
    suite = Suite(name="demo", target=_identity)
    suite.append(
        Scenario("scenario_1")
        .interact("hello")
        .checks(Equals(expected_value="hello", key="trace.last.outputs"))
    )
    return suite


@pytest.mark.asyncio
async def test_runner_logs_step_progress_at_info(caplog):
    """At INFO, running a suite surfaces scenario + check identity (start & end)."""
    with caplog.at_level(logging.INFO, logger="giskard"):
        result = await _demo_suite().run()

    assert result.pass_rate == 1.0

    messages = [r.getMessage() for r in caplog.records if r.name.startswith("giskard")]
    joined = "\n".join(messages)

    # Scenario identity is present so interleaved parallel output stays readable.
    assert "scenario_1" in joined
    # Check identity is surfaced (name falls back to kind, e.g. "equals").
    assert "equals" in joined
    # Both the start and the completion of the step are logged.
    assert any("running step" in m for m in messages)
    assert any("passed" in m for m in messages)


@pytest.mark.asyncio
async def test_runner_quiet_by_default(caplog):
    """At the default level the runner emits no INFO progress logs (backward-compatible)."""
    caplog.set_level(logging.WARNING, logger="giskard")

    await _demo_suite().run()

    below_warning = [
        r
        for r in caplog.records
        if r.name.startswith("giskard") and r.levelno < logging.WARNING
    ]
    assert below_warning == []
