from typing import Any

import pytest
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.llm.types import AssistantMessage, Choice, CompletionResponse
from giskard.scan.integrations.garak import _adapter
from giskard.scan.integrations.garak._adapter import (
    _resolve_detectors,
    _SkipMarker,
    garak_available,
)

pytestmark = pytest.mark.skipif(not garak_available(), reason="garak is not installed")


class _StubGenerator(BaseGenerator):
    async def _call_model(
        self,
        messages,
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        return CompletionResponse(
            choices=[Choice(message=AssistantMessage(content="Rating: [[1]]"))]
        )


@pytest.fixture(autouse=True)
def _default_generator(monkeypatch):
    from giskard.checks import settings as checks_settings

    monkeypatch.setattr(
        checks_settings, "get_default_generator", lambda: _StubGenerator()
    )
    # _adapter imports the name into its own namespace; patch there too if needed.
    monkeypatch.setattr(_adapter, "get_default_generator", lambda: _StubGenerator())


def test_judge_detector_gets_giskard_generator(monkeypatch):
    from garak.probes.base import Probe

    class _JudgeProbe(Probe):
        primary_detector = "judge.Refusal"
        # Match Probe's Iterable[str] annotation (list is invariant).
        extended_detectors = []

    detectors, skipped = _resolve_detectors(_JudgeProbe.__new__(_JudgeProbe), None)
    assert skipped == []
    assert len(detectors) == 1
    detector_label, detector = detectors[0]
    assert detector_label == "judge.Refusal"
    from giskard.scan.integrations.garak._judge_generator import GiskardJudgeGenerator

    # Judge detectors install evaluation_generator; base Detector has no such attr.
    assert isinstance(getattr(detector, "evaluation_generator"), GiskardJudgeGenerator)


def test_env_var_detector_without_key_is_skipped(monkeypatch):
    monkeypatch.delenv("PERSPECTIVE_API_KEY", raising=False)
    from garak.probes.base import Probe

    class _PerspectiveProbe(Probe):
        primary_detector = "perspective.Toxicity"
        extended_detectors = []

    detectors, skipped = _resolve_detectors(
        _PerspectiveProbe.__new__(_PerspectiveProbe), None
    )
    assert detectors == []
    assert len(skipped) == 1
    assert isinstance(skipped[0], _SkipMarker)
    assert "PERSPECTIVE_API_KEY" in skipped[0].reason


def test_detector_garak_exception_without_key_is_skipped(monkeypatch):
    """Non-key GarakException on load must yield a SkipMarker, not (None, None)."""
    import importlib

    from garak.exception import GarakException
    from garak.probes.base import Probe

    # Re-bind after block_garak_import may have purged sys.modules.
    live = importlib.import_module("giskard.scan.integrations.garak._adapter")

    class _BrokenProbe(Probe):
        primary_detector = "always.Fail"
        extended_detectors = []

    def _boom(name: str):
        raise GarakException("plugin broken")

    monkeypatch.setattr("garak._plugins.load_plugin", _boom)
    # Bypass judge short-circuit so load_plugin is exercised.
    monkeypatch.setattr(live, "_detector_class", lambda name: None)

    cache = live._DetectorCache(None)
    detector, marker = cache.get("always.Fail")
    assert detector is None
    assert marker is not None
    assert isinstance(marker, live._SkipMarker)
    assert marker.reason.startswith("load failed")
    assert "plugin broken" in marker.reason

    detectors, skipped = live._resolve_detectors(
        _BrokenProbe.__new__(_BrokenProbe), None
    )
    assert detectors == []
    assert len(skipped) == 1
    assert skipped[0].reason.startswith("load failed")


def test_detector_bare_exception_is_skipped(monkeypatch):
    """Bare Exception on load must yield a SkipMarker, not (None, None)."""
    import importlib

    from garak.probes.base import Probe

    live = importlib.import_module("giskard.scan.integrations.garak._adapter")

    class _BrokenProbe(Probe):
        primary_detector = "always.Fail"
        extended_detectors = []

    def _boom(name: str):
        raise RuntimeError("import exploded")

    monkeypatch.setattr("garak._plugins.load_plugin", _boom)
    monkeypatch.setattr(live, "_detector_class", lambda name: None)

    cache = live._DetectorCache(None)
    detector, marker = cache.get("always.Fail")
    assert detector is None
    assert marker is not None
    assert isinstance(marker, live._SkipMarker)
    assert marker.reason.startswith("load failed")
    assert "import exploded" in marker.reason

    detectors, skipped = live._resolve_detectors(
        _BrokenProbe.__new__(_BrokenProbe), None
    )
    assert detectors == []
    assert len(skipped) == 1


def test_skipped_detectors_emit_skip_results(monkeypatch):
    """Test that skip markers from _resolve_detectors are emitted as CheckResult.skip() per conversation."""
    from garak.attempt import Attempt, Conversation
    from garak.probes.base import Probe
    from giskard.scan.integrations.garak._adapter import GarakScanAdapter

    # Create a minimal probe with only a skipped detector
    monkeypatch.delenv("PERSPECTIVE_API_KEY", raising=False)

    class _SkipProbe(Probe):
        primary_detector = "perspective.Toxicity"
        extended_detectors = []

    detectors, skipped = _resolve_detectors(_SkipProbe.__new__(_SkipProbe), None)

    # Perspective detector should be skipped due to missing key
    assert len(detectors) == 0
    assert len(skipped) == 1
    assert isinstance(skipped[0], _SkipMarker)
    assert "PERSPECTIVE_API_KEY" in skipped[0].reason

    # Create a minimal attempt with 2 conversations (_evaluate_attempt only
    # needs len(conversations); empty Conversation stubs are enough).
    attempt = Attempt()
    attempt.conversations = [Conversation(turns=[]), Conversation(turns=[])]

    # Call _evaluate_attempt with the skip markers and no working detectors
    adapter = GarakScanAdapter()
    test_case_results = adapter._evaluate_attempt(attempt, detectors, skipped)

    # Should get 2 TestCaseResult objects (one per conversation)
    assert len(test_case_results) == 2

    for test_case_result in test_case_results:
        # Each should have exactly 1 CheckResult: the skip marker
        assert len(test_case_result.results) == 1

        # Result should be a skip result
        skip_result = test_case_result.results[0]
        assert skip_result.skipped, f"Expected skip status, got {skip_result.status}"
        assert "PERSPECTIVE_API_KEY" in str(skip_result.message)
        assert skip_result.details["detector"] == "perspective.Toxicity"
        assert skip_result.details["check_name"] == "perspective.Toxicity"
