"""Availability gate + kwarg-rejection + singleturn skip tests."""

import pytest
from giskard.scan.integrations.deepteam._adapter import (
    DeepTeamScanAdapter,
    deepteam_available,
)


def test_deepteam_available_is_bool():
    assert isinstance(deepteam_available(), bool)


@pytest.mark.skipif(not deepteam_available(), reason="deepteam not installed")
async def test_unexpected_kwarg_rejected():
    with pytest.raises(TypeError, match="unexpected keyword"):
        await DeepTeamScanAdapter().run(
            target=lambda x: x, description="d", probe="typo"
        )


@pytest.mark.skipif(not deepteam_available(), reason="deepteam not installed")
async def test_singleturn_with_only_multiturn_attacks_emits_skips():
    from giskard.checks import SuiteResult

    result = await DeepTeamScanAdapter().run(
        target=lambda x: x,
        description="d",
        attacks=["LinearJailbreaking"],
        vulnerabilities=["Bias"],
        target_mode="singleturn",
    )
    assert isinstance(result, SuiteResult)
    assert len(result.results) >= 1
    assert all(s.steps[0].results[0].skipped for s in result.results)
    assert any("LinearJailbreaking" in s.scenario_name for s in result.results)


@pytest.mark.skipif(not deepteam_available(), reason="deepteam not installed")
async def test_attacks_per_vulnerability_type_forwarded(monkeypatch):
    import deepteam

    captured: dict[str, object] = {}

    def fake_red_team(**kwargs: object) -> object:
        captured.update(kwargs)

        class _Empty:
            test_cases: list[object] = []

        return _Empty()

    monkeypatch.setattr(deepteam, "red_team", fake_red_team, raising=False)

    await DeepTeamScanAdapter().run(
        target=lambda x: "ok",
        description="d",
        vulnerabilities=["Bias"],
        attacks=["PromptInjection"],
        attacks_per_vulnerability_type=2,
        target_mode="singleturn",
    )
    assert captured["attacks_per_vulnerability_type"] == 2
