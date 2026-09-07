"""Tests for RTTestCase -> ScenarioResult translation."""

import pytest

pytest.importorskip("deepteam")

from giskard.checks import CheckResult
from giskard.scan.integrations.deepteam import _adapter
from giskard.scan.integrations.deepteam._bridge import ScanTargetCallback


class _TC:
    """Minimal RTTestCase stand-in with the fields the mapper reads."""

    def __init__(self, **kw):
        self.input = kw.get("input", "attack prompt")
        self.actual_output = kw.get("actual_output", "model reply")
        self.vulnerability = kw.get("vulnerability", "Bias")
        self.vulnerability_type = kw.get("vulnerability_type", "race")
        self.attack_method = kw.get("attack_method", "PromptInjection")
        self.risk_category = kw.get("risk_category", None)
        self.score = kw.get("score", 1.0)
        self.reason = kw.get("reason", "some reason")
        self.error = kw.get("error", None)
        self.turns = kw.get("turns", None)
        self.simulation_cost = kw.get("simulation_cost", None)
        self.evaluation_cost = kw.get("evaluation_cost", None)
        self.token_cost = kw.get("token_cost", None)
        self.retrieval_context = kw.get("retrieval_context", None)
        self.tools_called = kw.get("tools_called", None)


def _check(scenario) -> CheckResult:
    return scenario.steps[0].results[0]


async def test_resisted_score_maps_to_success():
    cb = ScanTargetCallback(target=lambda x: x)
    scenario = await _adapter._testcase_to_scenario(_TC(score=1.0), cb)
    # giskard.checks.CheckStatus values are "pass"/"fail"/"error"/"skip"; the
    # CheckResult.success() constructor sets status=CheckStatus.PASS.
    assert _check(scenario).status == "pass"


async def test_attack_success_maps_to_failure_polarity_flip():
    cb = ScanTargetCallback(target=lambda x: x)
    scenario = await _adapter._testcase_to_scenario(_TC(score=0.0), cb)
    assert _check(scenario).status == "fail"


async def test_error_field_maps_to_error():
    cb = ScanTargetCallback(target=lambda x: x)
    scenario = await _adapter._testcase_to_scenario(
        _TC(error="target blew up", score=None), cb
    )
    assert _check(scenario).status == "error"


async def test_missing_score_maps_to_skip_not_failure():
    """No score means no verdict: it must not be reported as a vulnerability hit."""
    cb = ScanTargetCallback(target=lambda x: x)
    scenario = await _adapter._testcase_to_scenario(_TC(score=None, reason=None), cb)
    check = _check(scenario)
    assert check.status == "skip"
    assert not scenario.failed
    assert scenario.skipped


async def test_details_carry_attack_method_and_risk_category():
    cb = ScanTargetCallback(target=lambda x: x)
    scenario = await _adapter._testcase_to_scenario(
        _TC(attack_method="Roleplay", risk_category="owasp:llm01"), cb
    )
    details = _check(scenario).details
    assert details["attack_method"] == "Roleplay"
    assert details["risk_category"] == "owasp:llm01"


async def test_scenario_name_and_tags():
    cb = ScanTargetCallback(target=lambda x: x)
    scenario = await _adapter._testcase_to_scenario(_TC(), cb)
    assert scenario.scenario_name.startswith("Bias/race · PromptInjection")
    assert "Bias" in scenario.tags


async def test_check_name_uses_vulnerability_and_type():
    cb = ScanTargetCallback(target=lambda x: x)
    scenario = await _adapter._testcase_to_scenario(_TC(), cb)
    check = _check(scenario)
    assert check.details["check_name"] == "Bias/race"
    assert check.metrics[0].name == "Bias/race"


async def test_check_name_resolves_enum_vulnerability_type():
    from enum import Enum

    class _BiasType(Enum):
        RACE = "race"

    cb = ScanTargetCallback(target=lambda x: x)
    scenario = await _adapter._testcase_to_scenario(
        _TC(vulnerability_type=_BiasType.RACE), cb
    )
    check = _check(scenario)
    assert check.details["check_name"] == "Bias/race"
    assert check.details["vulnerability_type"] == "race"
    assert scenario.scenario_name.startswith("Bias/race · PromptInjection")


async def test_scenario_name_resolves_real_deepteam_enum():
    from deepteam.vulnerabilities.misinformation.types import MisinformationType

    cb = ScanTargetCallback(target=lambda x: x)
    scenario = await _adapter._testcase_to_scenario(
        _TC(
            vulnerability="Misinformation",
            vulnerability_type=MisinformationType.EXPERTIZE_MISREPRESENTATION,
            attack_method="Linear Jailbreaking",
        ),
        cb,
    )
    assert (
        scenario.scenario_name
        == "Misinformation/expertize_misrepresentation · Linear Jailbreaking"
    )
    assert _check(scenario).details["check_name"] == (
        "Misinformation/expertize_misrepresentation"
    )
