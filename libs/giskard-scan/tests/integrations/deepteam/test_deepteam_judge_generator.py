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


class _FakeChat:
    """Stand-in for a giskard Chat holding a validated structured output."""

    def __init__(self, output):
        self.output = output


class _FakeWorkflow:
    """Fluent stand-in for the generator's chat/with_output/run workflow."""

    def __init__(self, prompt, output_by_schema):
        self.prompt = prompt
        self._output_by_schema = output_by_schema
        self._schema = None

    def with_output(self, schema, *args, **kwargs):
        self._schema = schema
        return self

    async def run(self, *args, **kwargs):
        # Return whatever the test registered for this schema, wrapped in a Chat.
        return _FakeChat(self._output_by_schema[self._schema])


class _FakeGenerator:
    """Minimal stand-in for a giskard BaseGenerator."""

    def __init__(self, output_by_schema=None):
        self.seen = []
        self.chat_prompts = []
        # schema class -> validated instance the fake workflow should return
        self._output_by_schema = output_by_schema or {}

    async def complete(self, messages, params=None, metadata=None):
        self.seen.append(messages)
        return _FakeResponse("fake-reply")

    def chat(self, prompt, *args, **kwargs):
        self.chat_prompts.append(prompt)
        return _FakeWorkflow(prompt, self._output_by_schema)


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


async def test_a_generate_honors_schema_returns_instance():
    # DeepEval calls a_generate(prompt, schema=X) and reads attributes off the
    # returned instance (e.g. ``res.data``). Regression guard: a bare-string
    # return used to raise "'str' object has no attribute 'data'".
    from deepteam.attacks.attack_simulator.schema import (
        SyntheticData,
        SyntheticDataList,
    )

    expected = SyntheticDataList(data=[SyntheticData(input="attack-1")])
    gen = _FakeGenerator(output_by_schema={SyntheticDataList: expected})
    llm = make_deepeval_llm(gen)

    res = await llm.a_generate("simulate attacks", schema=SyntheticDataList)

    assert isinstance(res, SyntheticDataList)
    assert res.data[0].input == "attack-1"  # attribute access DeepEval relies on
    # The prompt reached the structured-output workflow, not plain complete().
    assert gen.chat_prompts == ["simulate attacks"]
    assert gen.seen == []


def test_generate_sync_honors_schema_returns_instance():
    from deepteam.attacks.attack_simulator.schema import (
        SyntheticData,
        SyntheticDataList,
    )

    expected = SyntheticDataList(data=[SyntheticData(input="attack-1")])
    gen = _FakeGenerator(output_by_schema={SyntheticDataList: expected})
    llm = make_deepeval_llm(gen)

    res = llm.generate("simulate attacks", schema=SyntheticDataList)

    assert isinstance(res, SyntheticDataList)
    assert res.data[0].input == "attack-1"
