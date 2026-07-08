"""Tests for the Giskard-generator -> DeepEvalBaseLLM adapter."""

import pytest

pytest.importorskip("deepteam")

from deepeval.models import DeepEvalBaseLLM
from giskard.scan.integrations.deepteam._judge_generator import make_deepeval_llm


class _FakeMessage:
    def __init__(self, text):
        self.text = text


class _FakeChoice:
    def __init__(self, text):
        self.message = _FakeMessage(text)


class _FakeResponse:
    def __init__(self, text):
        self.choices = [_FakeChoice(text)]


class _FakeGenerator:
    """Minimal stand-in for a giskard BaseGenerator."""

    def __init__(self):
        self.seen = []

    async def complete(self, messages, params=None, metadata=None):
        self.seen.append(messages)
        return _FakeResponse("fake-reply")


def test_is_deepeval_base_llm_instance():
    llm = make_deepeval_llm(_FakeGenerator())
    assert isinstance(llm, DeepEvalBaseLLM)


def test_get_model_name_is_stable_str():
    llm = make_deepeval_llm(_FakeGenerator())
    assert isinstance(llm.get_model_name(), str)
    assert llm.get_model_name()  # non-empty


async def test_a_generate_routes_to_giskard_generator():
    gen = _FakeGenerator()
    llm = make_deepeval_llm(gen)
    out = await llm.a_generate("hello judge")
    assert out == "fake-reply"
    # The prompt was sent as a single user message.
    assert gen.seen and gen.seen[0][-1]["role"] == "user"
    assert gen.seen[0][-1]["content"] == "hello judge"


def test_generate_sync_routes_to_giskard_generator():
    gen = _FakeGenerator()
    llm = make_deepeval_llm(gen)
    assert llm.generate("hello judge") == "fake-reply"
