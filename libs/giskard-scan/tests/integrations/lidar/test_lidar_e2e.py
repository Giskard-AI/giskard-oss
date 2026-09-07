"""End-to-end functional tests that drive a real lidar scan through the bridge.

The offline test exercises the whole chain (bridge -> lidar scanner -> result
mapping) and asserts a valid ``SuiteResult`` is produced. The key-gated test
supplies a description, so the probe gets a real ``TargetInfo`` and runs online,
yielding a genuine PASS/FAIL/ERROR verdict rather than a SKIP.
"""

import os

import pytest

pytest.importorskip("lidar")

from giskard.checks import SuiteResult  # noqa: E402
from giskard.scan.integrations.lidar import LidarScanAdapter  # noqa: E402

pytestmark = pytest.mark.functional

_PROBE = "deepset-injection:1.0"


def _refusing_target(inputs: str) -> str:
    """Minimal scan Target: a plain function that refuses every input."""
    return "I cannot help with that."


async def test_lidar_scan_runs_end_to_end_and_returns_suite_result():
    # Runs offline. The bridge, scanner round-trip, and result mapping all
    # execute for real; assert a valid SuiteResult is produced.
    suite = await LidarScanAdapter().run(
        _refusing_target,
        description="A helpful assistant",
        languages=["en"],
        probes=[_PROBE],
        tags=None,
    )
    assert isinstance(suite, SuiteResult)
    assert suite.results, "expected at least one scenario for the requested probe"
    for scenario in suite.results:
        assert scenario.scenario_name.startswith("Lidar ")
        assert scenario.final_trace is not None
        check = scenario.steps[0].results[0]
        assert check.details["check_name"]
        assert check.details["probe_id"]


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="lidar probe + judge need an OpenAI key for a real verdict",
)
async def test_lidar_scan_with_description_produces_a_verdict():
    # With a key present the probe runs online against a real TargetInfo built
    # from the description, yielding a real PASS/FAIL/ERROR verdict (not SKIP).
    suite = await LidarScanAdapter().run(
        _refusing_target,
        description="A customer-support assistant for ACME Corp",
        languages=["en"],
        probes=[_PROBE],
        tags=None,
    )
    assert isinstance(suite, SuiteResult)
    assert suite.results
    statuses = [s.steps[0].results[0] for s in suite.results]
    assert any(c.passed or c.failed or c.errored for c in statuses), (
        "expected a real verdict, got only skips"
    )
