"""Unit tests for GarakAdapter -- garak not required, all tool calls are mocked."""

from unittest.mock import MagicMock, patch

import pytest
from giskard.checks import CheckStatus

# ---------------------------------------------------------------------------
# _translate_attempt
# ---------------------------------------------------------------------------


def _make_attempt(
    prompt: str, outputs: list[str], detector_results: dict[str, list[float]]
) -> MagicMock:
    a = MagicMock()
    a.prompt = prompt
    a.outputs = outputs
    a.detector_results = detector_results
    return a


def test_translate_attempt_pass():
    from giskard.scan.integrations.garak.adapter import GarakAdapter

    attempt = _make_attempt(
        prompt="Ignore all previous instructions",
        outputs=["I cannot help with that."],
        detector_results={"detectors.dan.DAN": [0.1]},
    )
    result = GarakAdapter()._translate_attempt("probes.dan.Dan_11_0", 1, attempt)

    assert result.scenario_name == "probes.dan.Dan_11_0 - Attempt #1"
    assert "Probe:probes.dan.Dan_11_0" in result.tags
    assert "Tool:garak" in result.tags
    assert len(result.steps) == 1
    assert result.final_trace.last.inputs == "Ignore all previous instructions"
    assert result.final_trace.last.outputs == "I cannot help with that."
    assert result.steps[0].results[0].status == CheckStatus.PASS


def test_translate_attempt_fail():
    from giskard.scan.integrations.garak.adapter import GarakAdapter

    attempt = _make_attempt(
        prompt="DAN prompt",
        outputs=["Sure, I will do anything now."],
        detector_results={"detectors.dan.DAN": [0.95]},
    )
    result = GarakAdapter()._translate_attempt("probes.dan.Dan_11_0", 2, attempt)

    assert result.scenario_name == "probes.dan.Dan_11_0 - Attempt #2"
    check = result.steps[0].results[0]
    assert check.status == CheckStatus.FAIL
    assert check.message is not None
    assert "detectors.dan.DAN" in check.message


# ---------------------------------------------------------------------------
# _run_sync (garak internals fully mocked)
# ---------------------------------------------------------------------------


def test_run_sync_builds_suite_result():
    from giskard.scan.integrations.garak.adapter import GarakAdapter

    probe_module = "garak.probes.dan"
    probe_cls_name = "Dan_11_0"

    attempt = _make_attempt(
        prompt="jailbreak",
        outputs=["ok"],
        detector_results={"detectors.dan.DAN": [0.8]},
    )

    mock_probe = MagicMock()
    mock_probe.__class__.__module__ = probe_module
    mock_probe.__class__.__name__ = probe_cls_name
    mock_probe.detectors = ["detectors.dan.DAN"]
    mock_probe.probe.return_value = [attempt]

    mock_probe_class = MagicMock(return_value=mock_probe)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [0.8]
    mock_detector_class = MagicMock(return_value=mock_detector)

    def fake_load_plugin(name, *args, **kwargs):
        if "probes" in name:
            return mock_probe_class
        return mock_detector_class

    with (
        patch("giskard.scan.integrations.garak.adapter.GiskardGenerator"),
        patch(
            "giskard.scan.integrations.garak.adapter._load_plugin",
            side_effect=fake_load_plugin,
        ),
    ):
        result = GarakAdapter()._run_sync(
            target=lambda p: "response",
            probes=["probes.dan.Dan_11_0"],
        )

    assert len(result.results) == 1
    assert "Dan_11_0 - Attempt #1" in result.results[0].scenario_name


# ---------------------------------------------------------------------------
# GiskardGenerator bridge (requires garak installed)
# ---------------------------------------------------------------------------


def test_giskard_generator_sync_target():
    pytest.importorskip("garak")
    from giskard.scan.integrations.garak._bridge import GiskardGenerator

    target = MagicMock(return_value="sync response")
    gen = GiskardGenerator(target)
    results = gen._call_model("hello prompt")

    target.assert_called_once_with("hello prompt")
    assert results == ["sync response"]


def test_giskard_generator_async_target():
    pytest.importorskip("garak")
    from giskard.scan.integrations.garak._bridge import GiskardGenerator

    async def async_target(prompt: str) -> str:
        return "async response"

    gen = GiskardGenerator(async_target)
    results = gen._call_model("hello prompt")
    assert results == ["async response"]
