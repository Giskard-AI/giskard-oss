from dataclasses import dataclass, field
from enum import Enum

import pytest

# Match the garak tests: real lidar messages, skip the module when lidar is absent.
pytest.importorskip("lidar")

from giskard.checks import SuiteResult  # noqa: E402
from giskard.scan.integrations.lidar._adapter import (  # noqa: E402
    _SEVERITY_SCORE,
    LidarScanAdapter,
)
from lidar.giskard_compat import make_message  # noqa: E402


# ProbeExecution / ProbeInfo / Attempt are still faked (hard to construct, like
# garak's _FakeProbe) — only the messages use the real lidar type via make_message.
class Severity(str, Enum):
    SAFE = "safe"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass
class FakeProbeInfo:
    id: str
    name: str
    tags: list[str] = field(default_factory=list)


@dataclass
class FakeAttempt:
    successful: bool
    messages: list = field(default_factory=list)
    severity: object = None
    reason: str = ""
    error: object = None
    metadata: dict = field(default_factory=dict)


@dataclass
class FakeResult:
    attempts: list


@dataclass
class FakeProbeExecution:
    probe_info: FakeProbeInfo
    result: object = None  # FakeResult, or None for errored/skipped probes
    status: str = "completed"
    error: object = None


@dataclass
class FakeScanResult:
    results: list


def _msgs():
    return [
        make_message(role="user", content="attack"),
        make_message(role="assistant", content="reply"),
    ]


async def test_successful_attempt_maps_to_failure():
    scan_result = FakeScanResult(
        results=[
            FakeProbeExecution(
                probe_info=FakeProbeInfo(
                    id="link-injection:1.0", name="Link Injection", tags=["injection"]
                ),
                result=FakeResult(
                    attempts=[
                        FakeAttempt(
                            successful=True,
                            messages=_msgs(),
                            severity=Severity.CRITICAL,
                            reason="leaked",
                        ),
                    ]
                ),
            )
        ]
    )
    suite = await LidarScanAdapter()._to_suite_result(scan_result, duration_ms=42)

    assert isinstance(suite, SuiteResult)
    assert suite.duration_ms == 42
    assert len(suite.results) == 1
    scenario = suite.results[0]
    assert scenario.scenario_name == "Lidar Link Injection #1"
    assert scenario.tags == ["injection"]

    check = scenario.steps[0].results[0]
    assert check.failed  # attack succeeded => failure
    assert check.message == "leaked"
    assert check.details["check_name"] == "Link Injection"
    assert check.details["probe_id"] == "link-injection:1.0"
    assert check.details["severity"] == "critical"
    assert check.metrics[0].name == "Link Injection"
    assert check.metrics[0].value == _SEVERITY_SCORE["critical"]


async def test_unsuccessful_attempt_maps_to_success():
    scan_result = FakeScanResult(
        results=[
            FakeProbeExecution(
                probe_info=FakeProbeInfo(id="pii-leak:1.0", name="PII Leak"),
                result=FakeResult(
                    attempts=[
                        FakeAttempt(
                            successful=False,
                            messages=_msgs(),
                            severity=Severity.SAFE,
                            reason="held",
                        ),
                    ]
                ),
            )
        ]
    )
    suite = await LidarScanAdapter()._to_suite_result(scan_result, duration_ms=1)
    check = suite.results[0].steps[0].results[0]
    assert check.passed


async def test_errored_attempt_maps_to_error():
    scan_result = FakeScanResult(
        results=[
            FakeProbeExecution(
                probe_info=FakeProbeInfo(id="x:1.0", name="X"),
                result=FakeResult(
                    attempts=[
                        FakeAttempt(
                            successful=False,
                            messages=[],
                            severity=None,
                            reason="boom",
                            error=object(),
                        ),
                    ]
                ),
            )
        ]
    )
    suite = await LidarScanAdapter()._to_suite_result(scan_result, duration_ms=1)
    check = suite.results[0].steps[0].results[0]
    assert check.errored


async def test_none_severity_omits_metric_and_label():
    scan_result = FakeScanResult(
        results=[
            FakeProbeExecution(
                probe_info=FakeProbeInfo(id="x:1.0", name="X"),
                result=FakeResult(
                    attempts=[
                        FakeAttempt(
                            successful=True, messages=_msgs(), severity=None, reason="r"
                        ),
                    ]
                ),
            )
        ]
    )
    suite = await LidarScanAdapter()._to_suite_result(scan_result, duration_ms=1)
    check = suite.results[0].steps[0].results[0]
    assert check.metrics == []
    assert "severity" not in check.details


async def test_unknown_severity_degrades_without_crashing():
    # A severity label with no score in _SEVERITY_SCORE (e.g. lidar adds a new
    # level) must not crash: the severity is still recorded, but no metric.
    class _FutureSeverity(str, Enum):
        BLOCKER = "blocker"

    scan_result = FakeScanResult(
        results=[
            FakeProbeExecution(
                probe_info=FakeProbeInfo(id="x:1.0", name="X"),
                result=FakeResult(
                    attempts=[
                        FakeAttempt(
                            successful=True,
                            messages=_msgs(),
                            severity=_FutureSeverity.BLOCKER,
                            reason="r",
                        ),
                    ]
                ),
            )
        ]
    )
    suite = await LidarScanAdapter()._to_suite_result(scan_result, duration_ms=1)
    check = suite.results[0].steps[0].results[0]
    assert check.metrics == []
    assert check.details["severity"] == "blocker"


async def test_attempt_metadata_carried_into_details():
    scan_result = FakeScanResult(
        results=[
            FakeProbeExecution(
                probe_info=FakeProbeInfo(id="x:1.0", name="X"),
                result=FakeResult(
                    attempts=[
                        FakeAttempt(
                            successful=True,
                            messages=_msgs(),
                            severity=Severity.MAJOR,
                            reason="r",
                            metadata={
                                "objective": "exfiltrate",
                                "injected_url": "http://x",
                            },
                        ),
                    ]
                ),
            )
        ]
    )
    suite = await LidarScanAdapter()._to_suite_result(scan_result, duration_ms=1)
    check = suite.results[0].steps[0].results[0]
    assert check.details["metadata"] == {
        "objective": "exfiltrate",
        "injected_url": "http://x",
    }


async def test_empty_metadata_omits_key():
    scan_result = FakeScanResult(
        results=[
            FakeProbeExecution(
                probe_info=FakeProbeInfo(id="x:1.0", name="X"),
                result=FakeResult(
                    attempts=[
                        FakeAttempt(
                            successful=False,
                            messages=_msgs(),
                            severity=Severity.SAFE,
                            reason="r",
                        ),
                    ]
                ),
            )
        ]
    )
    suite = await LidarScanAdapter()._to_suite_result(scan_result, duration_ms=1)
    check = suite.results[0].steps[0].results[0]
    assert "metadata" not in check.details


async def test_multiple_attempts_yield_numbered_scenarios():
    scan_result = FakeScanResult(
        results=[
            FakeProbeExecution(
                probe_info=FakeProbeInfo(id="x:1.0", name="X"),
                result=FakeResult(
                    attempts=[
                        FakeAttempt(
                            successful=True,
                            messages=_msgs(),
                            severity=Severity.MAJOR,
                            reason="1",
                        ),
                        FakeAttempt(
                            successful=False,
                            messages=_msgs(),
                            severity=Severity.SAFE,
                            reason="2",
                        ),
                    ]
                ),
            )
        ]
    )
    suite = await LidarScanAdapter()._to_suite_result(scan_result, duration_ms=1)
    names = [s.scenario_name for s in suite.results]
    assert names == ["Lidar X #1", "Lidar X #2"]


async def test_errored_probe_with_no_result_maps_to_error_scenario():
    # ProbeExecution with result=None (probe errored) -> one visible error
    # scenario, no crash on the missing .attempts.
    scan_result = FakeScanResult(
        results=[
            FakeProbeExecution(
                probe_info=FakeProbeInfo(id="boom:1.0", name="Boom"),
                result=None,
                status="errored",
                error="probe blew up",
            )
        ]
    )
    suite = await LidarScanAdapter()._to_suite_result(scan_result, duration_ms=1)
    assert len(suite.results) == 1
    scenario = suite.results[0]
    assert scenario.scenario_name == "Lidar Boom"
    # ScenarioResult.final_trace is a required Trace (no None allowed);
    # an errored probe never produced interactions, so expect an empty trace.
    assert scenario.final_trace.interactions == []
    check = scenario.steps[0].results[0]
    assert check.errored
    assert check.details["check_name"] == "Boom"
    assert check.details["probe_id"] == "boom:1.0"
