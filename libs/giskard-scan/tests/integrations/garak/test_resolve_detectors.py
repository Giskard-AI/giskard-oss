import pytest
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.llm.types import AssistantMessage, Choice, CompletionResponse
from giskard.scan.integrations.garak import _adapter
from giskard.scan.integrations.garak._adapter import _resolve_detectors, _SkipMarker
from giskard.scan.integrations.garak._judge_generator import GiskardJudgeGenerator


class _StubGenerator(BaseGenerator):
    async def _call_model(
        self,
        messages,
        params: GenerationParams,
        metadata: dict | None = None,
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


def _load_probe(name: str):
    _adapter._configure_garak()
    from garak._plugins import load_plugin

    return load_plugin(name)


def test_judge_detector_gets_giskard_generator(monkeypatch):
    from garak.probes.base import Probe

    class _JudgeProbe(Probe):
        primary_detector = "judge.Refusal"
        extended_detectors: list[str] = []

    detectors, skipped = _resolve_detectors(_JudgeProbe.__new__(_JudgeProbe), None)
    assert skipped == []
    assert len(detectors) == 1
    assert isinstance(detectors[0].evaluation_generator, GiskardJudgeGenerator)


def test_env_var_detector_without_key_is_skipped(monkeypatch):
    monkeypatch.delenv("PERSPECTIVE_API_KEY", raising=False)
    from garak.probes.base import Probe

    class _PerspectiveProbe(Probe):
        primary_detector = "perspective.Toxicity"
        extended_detectors: list[str] = []

    detectors, skipped = _resolve_detectors(
        _PerspectiveProbe.__new__(_PerspectiveProbe), None
    )
    assert detectors == []
    assert len(skipped) == 1
    assert isinstance(skipped[0], _SkipMarker)
    assert "PERSPECTIVE_API_KEY" in skipped[0].reason
