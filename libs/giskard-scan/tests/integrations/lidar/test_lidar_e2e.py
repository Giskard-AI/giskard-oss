"""End-to-end functional tests that drive a real lidar scan through the bridge.

The offline test exercises the whole chain (bridge -> lidar scanner -> result
mapping) without an LLM key: with ``discover_target_info=False`` lidar reports
the probe as SKIP, but the scan still runs and produces a valid ``SuiteResult``
with a rebuilt trace, which is what this test asserts.

The key-gated test enables target discovery and asserts a real probe verdict; it
runs only where an OpenAI key is available (lidar drives an LLM to profile the
target and to judge attempts).
"""

import os

import pytest

pytest.importorskip("lidar")

from giskard.checks import SuiteResult  # noqa: E402
from giskard.scan.integrations.lidar import LidarScanAdapter  # noqa: E402

pytestmark = pytest.mark.functional

# A cheap, deterministic static-payload probe; keeps runs fast and offline.
_PROBE = "deepset-injection:1.0"


def _refusing_target(inputs: str) -> str:
    """Minimal scan Target: a plain function that refuses every input."""
    return "I cannot help with that."


async def test_lidar_scan_runs_end_to_end_and_returns_suite_result():
    # Runs offline. With discovery off the probe SKIPs, but the bridge, the
    # scanner round-trip, and the result mapping all execute for real.
    suite = await LidarScanAdapter().run(_refusing_target, probes=[_PROBE], tags=None)
    assert isinstance(suite, SuiteResult)
    assert suite.results, "expected at least one scenario for the requested probe"
    for scenario in suite.results:
        assert scenario.scenario_name.startswith("Lidar ")
        # final_trace is a required field on ScenarioResult; the bridge always
        # supplies one (rebuilt from the attempt, or an empty Trace).
        assert scenario.final_trace is not None
        check = scenario.steps[0].results[0]
        assert check.details["check_name"]
        assert check.details["probe_id"]


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="lidar target discovery + judge need an OpenAI key",
)
async def test_lidar_scan_with_discovery_produces_a_verdict():
    # With a key present, target discovery succeeds and the probe actually runs,
    # yielding a real PASS/FAIL/ERROR verdict rather than a SKIP.
    adapter = LidarScanAdapter()
    suite = await adapter.run(_refusing_target, probes=[_PROBE], tags=None)
    assert isinstance(suite, SuiteResult)
    assert suite.results
    statuses = [s.steps[0].results[0] for s in suite.results]
    # At least one attempt produced a non-skip verdict (the refusing target
    # should mostly pass, but we only assert the chain reached a real verdict).
    assert any(c.passed or c.failed or c.errored for c in statuses)
