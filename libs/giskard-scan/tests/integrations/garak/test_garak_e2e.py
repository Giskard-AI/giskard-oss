"""Real end-to-end test: drive an actual garak probe through GarakScanAdapter.

Unlike test_garak_run.py (which injects a fake Probe/Detector to test only the
orchestration), this test runs a genuine garak probe against a stub target. It is
the test that catches the two failures the fake-probe tests cannot:

  #1 garak's global config must be loaded before probes are instantiated, or
     ``probe.probe()`` raises AttributeError on ``parallel_attempts`` / ``generations``;
  #2 ``generations`` must be pinned to 1, or garak fans out 5 responses per prompt
     (5x scenarios and, for multiturn probes, a polluted Trace).

We use ``probes.goodside.ThreatenJSON``: a single-turn probe with exactly one prompt
whose detector (``goodside.PlainJSON``) is pure string inspection — no model calls,
no network, no weight downloads — so the test is deterministic and fast (~0.2s).

Marked ``functional``: it runs whenever garak is installed and is skipped otherwise.
"""

import pytest

pytest.importorskip("garak")

from typing import cast

from garak._plugins import load_plugin
from garak.probes.base import Probe
from giskard.checks import SuiteResult
from giskard.scan.integrations.garak._adapter import (
    GarakScanAdapter,
    _resolve_probes,
    garak_available,
)

pytestmark = pytest.mark.functional

_PROBE = "probes.goodside.ThreatenJSON"


@pytest.mark.skipif(not garak_available(), reason="garak is not installed")
def test_resolve_probes_empty_list_returns_no_probes() -> None:
    probes, skipped = _resolve_probes([])
    assert probes == []
    assert skipped == []


@pytest.mark.skipif(not garak_available(), reason="garak is not installed")
async def test_run_actual_garak_probe() -> None:
    calls: list[str] = []

    def target(inputs: str) -> str:
        calls.append(inputs)
        return "I will not comply."

    result = await GarakScanAdapter().run(target=target, probes=[_PROBE])

    # #1: a real probe ran to completion without garak-config AttributeErrors.
    assert isinstance(result, SuiteResult)
    assert len(result.results) >= 1

    # #2: generations pinned to 1 -> the single-prompt probe hits the target once,
    # not garak's default of five times, and yields exactly one scenario.
    assert len(calls) == 1
    assert len(result.results) == 1

    scenario = result.results[0]
    # The scenario name reflects a single run (no "— run 5" fan-out).
    assert "ThreatenJSON" in scenario.scenario_name
    assert scenario.scenario_name.endswith("#1")

    # The stub response was recorded on a clean, single-turn trace.
    assert len(scenario.final_trace.interactions) == 1
    assert scenario.final_trace.last is not None
    assert scenario.final_trace.last.outputs == "I will not comply."

    # A detector actually scored the attempt, producing one check result.
    assert len(scenario.steps) == 1
    assert len(scenario.steps[0].results) >= 1

    probe = cast("Probe", load_plugin(_PROBE))
    assert scenario.tags == list(probe.tags)
