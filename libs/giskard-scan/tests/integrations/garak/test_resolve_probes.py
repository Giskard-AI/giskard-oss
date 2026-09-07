"""Tests for skip markers when requested garak probes cannot run."""

import pytest

pytest.importorskip("garak")

from giskard.scan.integrations.garak._adapter import (
    GarakScanAdapter,
    _resolve_probes,
    _SkipMarker,
    garak_available,
)

pytestmark = pytest.mark.skipif(not garak_available(), reason="garak is not installed")

_INACTIVE = "probes.lmrc.Profanity"
_UNKNOWN = "probes.definitely.NotARealProbe"
_ACTIVE = "probes.goodside.ThreatenJSON"


def test_resolve_probes_empty_list_returns_no_probes_or_skips() -> None:
    probes, skipped = _resolve_probes([])
    assert probes == []
    assert skipped == []


def test_resolve_probes_none_uses_curated_defaults() -> None:
    from giskard.scan.integrations.garak._adapter import DEFAULT_PROBES

    probes, skipped = _resolve_probes(None)
    assert skipped == []
    assert len(probes) == len(DEFAULT_PROBES)


def test_default_probes_align_with_deepteam_themes() -> None:
    from giskard.scan.integrations.garak._adapter import DEFAULT_PROBES

    assert DEFAULT_PROBES == [
        "probes.lmrc.SlurUsage",
        "probes.goodside.WhoIsRiley",
        "probes.lmrc.QuackMedicine",
        "probes.goodside.ThreatenJSON",
        "probes.dan.AutoDANCached",
        "probes.web_injection.StringAssemblyDataExfil",
    ]


def test_resolve_probes_all_is_broader_than_defaults() -> None:
    from giskard.scan.integrations.garak._adapter import DEFAULT_PROBES

    curated, _ = _resolve_probes(None)
    all_probes, _ = _resolve_probes("all")
    assert len(all_probes) > len(curated)
    assert len(all_probes) > len(DEFAULT_PROBES)


def test_list_probes_includes_threaten_json() -> None:
    from giskard.scan.integrations.garak import list_probes

    names = list_probes()
    assert "probes.goodside.ThreatenJSON" in names
    inactive = list_probes(include_inactive=True)
    assert "probes.lmrc.Profanity" in inactive
    assert "probes.lmrc.Profanity" not in names


def test_resolve_unknown_probe_is_skipped() -> None:
    probes, skipped = _resolve_probes([_UNKNOWN])
    assert probes == []
    assert len(skipped) == 1
    assert isinstance(skipped[0], _SkipMarker)
    assert skipped[0].name == _UNKNOWN
    assert skipped[0].reason == "unknown"


def test_resolve_inactive_probe_is_skipped() -> None:
    probes, skipped = _resolve_probes([_INACTIVE])
    assert probes == []
    assert len(skipped) == 1
    assert skipped[0].name == _INACTIVE
    assert skipped[0].reason == "inactive"


def test_resolve_mixes_active_probe_with_skips() -> None:
    probes, skipped = _resolve_probes([_ACTIVE, _UNKNOWN, _INACTIVE])
    assert len(probes) == 1
    assert (
        probes[0].probename.endswith("ThreatenJSON")
        or "ThreatenJSON" in probes[0].probename
    )
    assert {s.name: s.reason for s in skipped} == {
        _UNKNOWN: "unknown",
        _INACTIVE: "inactive",
    }


def test_resolve_load_failure_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    import garak._plugins as plugins

    monkeypatch.setattr(
        plugins, "enumerate_plugins", lambda category: [(_ACTIVE, True)]
    )
    monkeypatch.setattr(
        plugins, "load_plugin", lambda name: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    probes, skipped = _resolve_probes([_ACTIVE])
    assert probes == []
    assert len(skipped) == 1
    assert skipped[0].name == _ACTIVE
    assert skipped[0].reason.startswith("load failed")


def test_resolve_probes_all_records_load_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import garak._plugins as plugins

    good = "probes.goodside.ThreatenJSON"
    bad = "probes.lmrc.SlurUsage"

    monkeypatch.setattr(
        plugins,
        "enumerate_plugins",
        lambda category: [(good, True), (bad, True)],
    )

    def _load(name: str):
        if name == bad:
            raise RuntimeError("boom")
        # Minimal active probe stub — only ``active`` is read after load.
        probe = type("Probe", (), {"active": True, "probename": name})()
        return probe

    monkeypatch.setattr(plugins, "load_plugin", _load)

    probes, skipped = _resolve_probes("all")
    assert len(probes) == 1
    assert probes[0].probename == good
    assert len(skipped) == 1
    assert skipped[0].name == bad
    assert skipped[0].reason.startswith("load failed")


async def test_run_emits_skip_scenario_for_missing_probes() -> None:
    def target(inputs: str) -> str:
        return "ok"

    result = await GarakScanAdapter().run(
        target=target,
        probes=[_UNKNOWN, _INACTIVE, _ACTIVE],
        target_mode="singleturn",
    )

    skip_by_name = {
        scenario.scenario_name: scenario.steps[0].results[0]
        for scenario in result.results
        if scenario.steps and scenario.steps[0].results[0].skipped
    }
    assert "probes.definitely.NotARealProbe (skipped)" in skip_by_name or any(
        "NotARealProbe" in k for k in skip_by_name
    )
    unknown_key = next(k for k in skip_by_name if "NotARealProbe" in k)
    assert "not a known" in (skip_by_name[unknown_key].message or "").lower()
    inactive_key = next(k for k in skip_by_name if "Profanity" in k)
    assert "inactive" in (skip_by_name[inactive_key].message or "").lower()

    # Active probe still produced a real (non-skip-only) scenario.
    active = [
        s
        for s in result.results
        if not (s.steps and s.steps[0].results and s.steps[0].results[0].skipped)
    ]
    assert len(active) >= 1
