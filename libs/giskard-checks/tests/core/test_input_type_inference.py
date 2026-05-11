from collections.abc import AsyncGenerator
from typing import Any, Literal, override

import pytest
from giskard.checks import Interact, Trace
from giskard.checks.core.input_generator import InputGenerator
from giskard.checks.core.interaction.interact import _infer_input_type
from pydantic import BaseModel

# --- _infer_input_type unit tests ---


class MyModel(BaseModel):
    role: Literal["user"] = "user"
    content: str


def test_infer_returns_none_for_non_callable():
    assert _infer_input_type("static value") is None


def test_infer_returns_none_for_callable_with_no_annotation():
    assert _infer_input_type(lambda x: x) is None


def test_infer_returns_none_for_str_annotated_callable():
    def target(input: str) -> str:
        return input

    assert _infer_input_type(target) is None


def test_infer_returns_base_model_type():
    def target(input: MyModel) -> str:
        return input.content

    assert _infer_input_type(target) is MyModel


def test_infer_returns_none_for_forward_ref_that_cannot_resolve():
    def target(input: "UnresolvableType") -> str:  # noqa: F821 # pyright: ignore[reportUndefinedVariable]
        return str(input)

    assert _infer_input_type(target) is None


def test_infer_returns_base_model_type_for_callable_class():
    class AgentAdapter:
        def __call__(self, input: MyModel) -> str:
            return input.content

    assert _infer_input_type(AgentAdapter()) is MyModel


def test_infer_returns_none_for_callable_class_with_str_annotation():
    class AgentAdapter:
        def __call__(self, input: str) -> str:
            return input

    assert _infer_input_type(AgentAdapter()) is None


# --- Integration: Interact forwards input_type to InputGenerator ---


class RecordingTrace(Trace[str, str], frozen=True):
    def _repr_prompt_(self) -> str:
        return ""


@InputGenerator.register("recording_generator")
class RecordingGenerator(InputGenerator[RecordingTrace]):
    received_input_type: type | None = None

    @override
    async def __call__(
        self, trace, input_type=None
    ) -> AsyncGenerator[Any, RecordingTrace]:
        self.received_input_type = input_type
        yield "hello"


@pytest.mark.asyncio
async def test_interact_forwards_base_model_input_type_to_generator():
    gen = RecordingGenerator()

    def target(inputs: MyModel) -> str:
        return str(inputs)

    interact = Interact(inputs=gen, outputs=target)
    trace = RecordingTrace()
    agen = interact.generate(trace)
    await anext(agen)
    assert gen.received_input_type is MyModel


@pytest.mark.asyncio
async def test_interact_passes_none_input_type_for_str_annotated_target():
    gen = RecordingGenerator()

    def target(inputs: str) -> str:
        return inputs

    interact = Interact(inputs=gen, outputs=target)
    trace = RecordingTrace()
    agen = interact.generate(trace)
    await anext(agen)
    assert gen.received_input_type is None
