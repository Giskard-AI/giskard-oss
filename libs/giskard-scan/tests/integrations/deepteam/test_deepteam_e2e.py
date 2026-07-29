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
        self.turns: list[object] | None = None
        self.simulation_cost = None
        self.evaluation_cost = None
        self.token_cost = None
        self.retrieval_context = None
        self.tools_called = None


class _FakeRiskAssessment:
    test_cases = [_FakeRTTestCase()]


async def test_third_party_scan_deepteam_end_to_end(monkeypatch):
    import asyncio

    import deepteam

    def fake_red_team(**kwargs):
        # Drive the real bridge callback once so the uuid cache is populated the
        # way a real run would, then hand back a test case whose turns carry the
        # stamped assistant RTTurn. This exercises the true
        # entry-point -> adapter -> callback -> Trace -> _trace_for path, not just
        # a static fixture.
        callback = kwargs["model_callback"]
        # fake_red_team runs inside asyncio.to_thread (a worker thread with no
        # running loop), so asyncio.run is the right primitive to drive the
        # async callback here.
        turn = asyncio.run(callback("attack"))
        case = _FakeRTTestCase()
        case.turns = [turn]
        assessment = _FakeRiskAssessment()
        assessment.test_cases = [case]
        return assessment

    # The adapter binds red_team via `from deepteam import red_team` inside run(),
    # so patch the module attribute it will resolve at call time.
    monkeypatch.setattr(deepteam, "red_team", fake_red_team, raising=False)

    def target(inputs: str) -> str:
        return "refusal"

    result = await third_party_scan(
        target=target,
        tool="deepteam",
        description="a support agent",
        attacks=["PromptInjection"],
        vulnerabilities=["Bias"],
    )

    assert isinstance(result, SuiteResult)
    assert len(result.results) == 1
    assert result.results[0].scenario_name.startswith("Bias/race")
    assert result.results[0].steps[0].results[0].status == "pass"
