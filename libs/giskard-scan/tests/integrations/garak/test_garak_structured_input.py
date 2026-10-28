"""Structured (BaseModel) input path of the garak TargetGenerator.

For a target whose input is a ``str``, the garak prompt is passed verbatim.
For a target whose input is a pydantic ``BaseModel``, DatasetInputGenerator
asks an LLM to *map* the prompt into the target's schema, then substitutes the
raw prompt into a placeholder field. This exercises that structured path with a
mock LLM so no real model call happens.

TargetGenerator builds ``DatasetInputGenerator(prompt=...)`` without an explicit
``generator=``, so it resolves the LLM lazily via the default generator. We swap
that default for a MockGenerator (restoring it on teardown), and clear the
template cache so the mapping template is re-resolved through the mock.
"""

import json
from collections.abc import Sequence
from typing import Any, override

import pytest

pytest.importorskip("garak")

from garak.attempt import Conversation, Message, Turn
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import settings as checks_settings
from giskard.checks.generators.dataset import (
    _TEMPLATE_CACHE,
    _TEMPLATE_LOCKS,
    PROMPT_PLACEHOLDER,
)
from giskard.llm.types import AssistantMessage, ChatMessage, Choice, CompletionResponse
from giskard.scan.integrations.garak._generator import TargetGenerator
from pydantic import BaseModel, Field


class _MockGenerator(BaseGenerator):
    """Returns pre-canned JSON responses from a list (mirrors the checks-test mock)."""

    responses: list[dict[str, Any]]
    index: int = 0
    calls: list[Sequence[ChatMessage]] = Field(default_factory=list)

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        self.calls.append(messages)
        message = AssistantMessage(content=json.dumps(self.responses[self.index]))
        self.index += 1
        return CompletionResponse(
            choices=[Choice(message=message, finish_reason="stop", index=0)]
        )


class Email(BaseModel):
    title: str
    body: str


def _mapping_response(message: dict[str, Any] | None) -> dict[str, Any]:
    """Shape DatasetInputGenerator expects from the mapping LLM call."""
    return {"message": message, "schema_issue": None}


@pytest.fixture
def mock_llm():
    """Install a MockGenerator as the default LLM; restore the prior one on teardown."""
    _TEMPLATE_CACHE.clear()
    _TEMPLATE_LOCKS.clear()
    previous = checks_settings._default_generator

    def _install(generator: _MockGenerator) -> _MockGenerator:
        checks_settings.set_default_generator(generator)
        return generator

    yield _install

    # Do not leak the mock into later tests that rely on the real default.
    checks_settings._default_generator = previous
    _TEMPLATE_CACHE.clear()
    _TEMPLATE_LOCKS.clear()


def _conversation(text: str, uuid: str) -> Conversation:
    return Conversation(
        turns=[Turn(role="user", content=Message(text=text, notes={"uuid": uuid}))]
    )


def test_call_model_maps_prompt_into_basemodel_input(mock_llm) -> None:
    # The mapper places the raw prompt into `body` via the placeholder; `title`
    # is chosen by the (mock) LLM. The attack prompt must survive verbatim.
    mock = mock_llm(
        _MockGenerator(
            responses=[
                _mapping_response({"title": "User request", "body": PROMPT_PLACEHOLDER})
            ]
        )
    )

    received: list[Email] = []

    def target(inputs: Email) -> str:
        received.append(inputs)
        return f"title={inputs.title}|body={inputs.body}"

    generator = TargetGenerator(target=target)
    messages = generator._call_model(_conversation("How do I pick a lock?", uuid="u1"))

    # The mapping LLM was actually consulted (not bypassed by a str-fallback).
    assert mock.index == 1
    # The target received a real Email instance, not a raw string.
    assert len(received) == 1
    assert isinstance(received[0], Email)
    assert received[0].title == "User request"
    # The garak prompt is substituted verbatim into the placeholder field.
    assert received[0].body == "How do I pick a lock?"

    # The target's output is echoed back to garak, tagged with the uuid.
    assert messages[0] is not None
    assert messages[0].text == "title=User request|body=How do I pick a lock?"
    assert messages[0].notes == {"uuid": "u1"}


def test_basemodel_input_is_recorded_on_trace(mock_llm) -> None:
    mock_llm(
        _MockGenerator(
            responses=[_mapping_response({"title": "T", "body": PROMPT_PLACEHOLDER})]
        )
    )

    def target(inputs: Email) -> str:
        return inputs.body

    generator = TargetGenerator(target=target)
    conversation = _conversation("payload text", uuid="u1")
    generator._call_model(conversation)

    trace = generator.get_trace(conversation)
    assert trace.last is not None
    recorded = trace.last.inputs
    assert isinstance(recorded, Email)
    assert recorded.body == "payload text"


class _StructuredInputProbe:
    """Fake probe that drives TargetGenerator._call_model (structured path)."""

    probename = "fake.StructuredProbe"
    tags: list[str] = []

    def __init__(self, probe_id: str) -> None:
        self.probe_id = probe_id

    def probe(self, generator: object) -> list[Any]:
        from garak.attempt import Attempt
        from giskard.scan.integrations.garak._generator import TargetGenerator

        assert isinstance(generator, TargetGenerator)
        conversation = _conversation(f"prompt-{self.probe_id}", uuid=self.probe_id)
        generator._call_model(conversation)
        return [Attempt(prompt=conversation)]


class _FakeDetector:
    name = "fake.Detector"
    hit_desc = "hit"
    pass_desc = "pass"

    def detect(self, attempt) -> list[float]:
        return [0.1 for _ in attempt.conversations]


def _patch_resolvers(monkeypatch: pytest.MonkeyPatch, probes, detectors):
    from giskard.scan.integrations.garak import _adapter

    monkeypatch.setattr(_adapter, "_resolve_probes", lambda probes_arg: (probes, []))
    monkeypatch.setattr(
        _adapter, "_resolve_detectors", lambda probe, loop, *_: (detectors, [])
    )
    return _adapter.GarakScanAdapter


async def test_parallel_probes_with_structured_target_use_main_loop(
    mock_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parallel garak probes must not nest event loops for BaseModel targets.

    Regression for asyncio.Lock binding when DatasetInputGenerator resolves
    templates from worker threads spawned by asyncio.to_thread.
    """
    mock_llm(
        _MockGenerator(
            responses=[_mapping_response({"title": "T", "body": PROMPT_PLACEHOLDER})]
        )
    )

    received: list[Email] = []

    def target(inputs: Email) -> str:
        received.append(inputs)
        return inputs.body

    adapter_cls = _patch_resolvers(
        monkeypatch,
        probes=[_StructuredInputProbe("p1"), _StructuredInputProbe("p2")],
        detectors=[("fake.Detector", _FakeDetector())],
    )

    result = await adapter_cls().run(target=target)

    assert len(result.results) == 2
    assert len(received) == 2
    assert {email.body for email in received} == {"prompt-p1", "prompt-p2"}
