"""target_mode='singleturn' drops lidar's multi-turn probes.

Lidar exposes no structural "multiturn" marker (no shared base class), so the
adapter filters on the ``gsk:probe-type='multi-turn'`` tag. The guard test pins
that tag string against a real multi-turn probe so a lidar rename fails loudly
here instead of silently leaking multi-turn probes into single-turn scans.
"""

import pytest

# Real lidar registry/probes; skip the module when lidar is absent.
pytest.importorskip("lidar")

from giskard.checks import SuiteResult  # noqa: E402
from giskard.scan.integrations.lidar._adapter import (  # noqa: E402
    _MULTITURN_TAG,
    LidarScanAdapter,
    _singleturn_probe_ids,
)
from lidar.utils.probe_registry import ProbeRegistry  # noqa: E402


def test_multiturn_tag_still_marks_a_known_multiturn_probe():
    # Guard: if lidar renames the tag, crescendo stops carrying _MULTITURN_TAG
    # and this fails — surfacing the break instead of silently running crescendo
    # under target_mode='singleturn'.
    (crescendo,) = ProbeRegistry().get_probes(None, probe_ids=["crescendo:1.0"])
    assert _MULTITURN_TAG in crescendo.info().tags


def test_singleturn_drops_multiturn_probes_keeps_singleturn():
    ids = _singleturn_probe_ids(
        probes=["crescendo:1.0", "goat:1.0", "deepset-injection:1.0"], tags=None
    )
    # crescendo + goat are multi-turn -> dropped; deepset is single-turn -> kept.
    assert ids == ["deepset-injection:1.0"]


def test_singleturn_over_only_multiturn_probes_yields_empty():
    ids = _singleturn_probe_ids(probes=["crescendo:1.0", "goat:1.0"], tags=None)
    assert ids == []


@pytest.fixture
def target():
    def _target(inputs: str) -> str:
        return "ok"

    return _target


def _patch_run_scan(monkeypatch, fake):
    import lidar

    monkeypatch.setattr(lidar, "run_scan", fake)


async def test_run_singleturn_passes_filtered_ids_and_no_tag_filter(
    monkeypatch, target
):
    captured = {}

    class _FakeScanRun:
        scan_result = None

        async def wait_for_completion(self):
            return None

    async def fake_run_scan(**kwargs):
        captured.update(kwargs)
        return _FakeScanRun()

    _patch_run_scan(monkeypatch, fake_run_scan)

    await LidarScanAdapter().run(
        target,
        description="A bot",
        probes=["crescendo:1.0", "deepset-injection:1.0"],
        target_mode="singleturn",
    )

    # Only the single-turn probe reaches run_scan, and the tag filter is cleared
    # because filtering already happened during id resolution.
    assert captured["probe_ids"] == ["deepset-injection:1.0"]
    assert captured["tags_filter"] is None


async def test_run_singleturn_all_multiturn_returns_empty_suite_without_scanning(
    monkeypatch, target
):
    called = False

    async def fake_run_scan(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("run_scan must not be called when no probes remain")

    _patch_run_scan(monkeypatch, fake_run_scan)

    suite = await LidarScanAdapter().run(
        target,
        description="A bot",
        probes=["crescendo:1.0", "goat:1.0"],
        target_mode="singleturn",
    )

    assert isinstance(suite, SuiteResult)
    assert suite.results == []
    assert called is False


async def test_run_multiturn_default_passes_probes_and_tags_through(
    monkeypatch, target
):
    # The default (no target_mode / "multiturn") path is unchanged: probes and
    # tags pass straight to run_scan, no registry resolution.
    captured = {}

    class _FakeScanRun:
        scan_result = None

        async def wait_for_completion(self):
            return None

    async def fake_run_scan(**kwargs):
        captured.update(kwargs)
        return _FakeScanRun()

    _patch_run_scan(monkeypatch, fake_run_scan)

    await LidarScanAdapter().run(
        target,
        description="A bot",
        probes=["crescendo:1.0"],
        tags=["some-tag"],
    )

    assert captured["probe_ids"] == ["crescendo:1.0"]
    assert captured["tags_filter"] == ["some-tag"]
