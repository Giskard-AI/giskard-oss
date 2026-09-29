from types import SimpleNamespace

import pytest
from giskard.scan.integrations.lidar import _adapter


def test_require_lidar_raises_private_message(monkeypatch):
    monkeypatch.setattr(_adapter, "find_spec", lambda name: None)
    with pytest.raises(ImportError) as exc_info:
        _adapter._require_lidar()
    message = str(exc_info.value)
    assert "private" in message.lower()
    assert "git+https://github.com/Giskard-AI/lidar.git@v0.2.7" in message


def test_lidar_available_reflects_find_spec(monkeypatch):
    monkeypatch.setattr(_adapter, "find_spec", lambda name: object())
    assert _adapter.lidar_available() is True
    monkeypatch.setattr(_adapter, "find_spec", lambda name: None)
    assert _adapter.lidar_available() is False


def _attempt(successful):
    """A duck-typed lidar attempt; the real one is not constructible without lidar."""
    return SimpleNamespace(
        successful=successful, error=None, severity=None, reason="", metadata={}
    )


_PROBE = SimpleNamespace(id="pii-leak:1.0", name="PII Leak", tags=[])


def test_successful_attempt_maps_to_failure():
    check = _adapter.LidarScanAdapter()._attempt_to_check(_PROBE, _attempt(True))
    assert check.failed


def test_unsuccessful_attempt_maps_to_success():
    check = _adapter.LidarScanAdapter()._attempt_to_check(_PROBE, _attempt(False))
    assert check.passed


def test_attempt_without_verdict_is_skipped_not_passed():
    # lidar reached no verdict: reporting PASS would be indistinguishable from a
    # probe that actually held, which is what the deepteam and garak adapters
    # already avoid.
    check = _adapter.LidarScanAdapter()._attempt_to_check(_PROBE, _attempt(None))
    assert check.skipped
    assert not check.passed
    assert not check.failed
    assert check.message == "lidar returned no verdict"


def test_attempt_without_verdict_keeps_probe_reason():
    attempt = _attempt(None)
    attempt.reason = "judge gave up"
    check = _adapter.LidarScanAdapter()._attempt_to_check(_PROBE, attempt)
    assert check.skipped
    assert check.message == "judge gave up"


def test_errored_attempt_still_maps_to_error():
    # The error branch runs before the verdict branch: an attempt that carries
    # both an error and no verdict stays an error.
    attempt = _attempt(None)
    attempt.error = RuntimeError("boom")
    attempt.reason = "boom"
    check = _adapter.LidarScanAdapter()._attempt_to_check(_PROBE, attempt)
    assert check.errored
