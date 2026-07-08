"""End-to-end: third_party_scan(tool='deepteam') against a fake target.

Uses a stubbed deepteam.red_team so the test runs without real LLM calls but
still exercises the full adapter + entry-point + translation path.
"""

import pytest

pytest.importorskip("deepteam")

from giskard.checks import SuiteResult
from giskard.scan.integrations import third_party_scan


class _FakeRTTestCase:
    def __init__(self):
        self.input = "attack"
        self.actual_output = "refusal"
        self.vulnerability = "Bias"
        self.vulnerability_type = "race"
        self.attack_method = "PromptInjection"
        self.risk_category = None
        self.score = 1.0
        self.reason = "resisted"
        self.error = None
        self.turns = None
        self.simulation_cost = None
        self.evaluation_cost = None
        self.token_cost = None
        self.retrieval_context = None
        self.tools_called = None


class _FakeRiskAssessment:
    test_cases = [_FakeRTTestCase()]


async def test_third_party_scan_deepteam_end_to_end(monkeypatch):
    import deepteam

    def fake_red_team(**kwargs):
        # Drive the callback once so the uuid cache is populated like a real run.
        return _FakeRiskAssessment()

    monkeypatch.setattr(deepteam, "red_team", fake_red_team)
    # The adapter imports red_team via `from deepteam import red_team`, so patch
    # the name it will bind. Patch the module attribute BEFORE the call.
    monkeypatch.setattr("deepteam.red_team", fake_red_team, raising=False)

    result = await third_party_scan(
        target=lambda x: "refusal",
        tool="deepteam",
        description="a support agent",
        attacks=["PromptInjection"],
        vulnerabilities=["Bias"],
    )

    assert isinstance(result, SuiteResult)
    assert len(result.results) == 1
    assert result.results[0].scenario_name.startswith("DeepTeam Bias/race")
    assert result.results[0].steps[0].results[0].status == "pass"
