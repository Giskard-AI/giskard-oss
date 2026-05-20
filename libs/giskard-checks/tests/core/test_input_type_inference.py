from collections.abc import AsyncGenerator
from typing import Any, Literal, override

import pytest
from giskard.checks import Interact, Interaction, Trace
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


def test_infer_returns_str_for_str_annotated_callable():
    def target(input: str) -> str:
        return input

    assert _infer_input_type(target) is str


def test_infer_returns_base_model_type():
    def target(input: MyModel) -> str:
        return input.content

    assert _infer_input_type(target) is MyModel


def test_infer_returns_list_type():
    def target(input: list[str]) -> str:
        return ", ".join(input)

    assert _infer_input_type(target) == list[str]


def test_infer_returns_int_type():
    def target(input: int) -> str:
        return str(input)

    assert _infer_input_type(target) is int


def test_infer_returns_dict_type():
    def target(input: dict[str, Any]) -> str:
        return str(input)

    assert _infer_input_type(target) == dict[str, Any]


def test_infer_returns_optional_base_model_type():
    def target(input: MyModel | None) -> str:
        return "" if input is None else input.content

    assert _infer_input_type(target) == MyModel | None


def test_infer_returns_none_for_forward_ref_that_cannot_resolve():
    def target(input: "UnresolvableType") -> str:  # noqa: F821 # pyright: ignore[reportUndefinedVariable]
        return str(input)

    assert _infer_input_type(target) is None


def test_infer_returns_base_model_type_for_callable_class():
    class AgentAdapter:
        def __call__(self, input: MyModel) -> str:
            return input.content

    assert _infer_input_type(AgentAdapter()) is MyModel


def test_infer_returns_str_for_callable_class_with_str_annotation():
    class AgentAdapter:
        def __call__(self, input: str) -> str:
            return input

    assert _infer_input_type(AgentAdapter()) is str


def test_infer_returns_none_for_pydantic_incompatible_type():
    def target(input: object) -> str:  # type: ignore[misc]
        return str(input)

    # get_type_hints returns `object` which TypeAdapter handles fine;
    # use an actual instance (not a type) to trigger a TypeError path.
    class NotAType:
        pass

    not_a_type_instance = NotAType()

    # Patch first_param_type to be an instance rather than a type to verify
    # the TypeError branch: call _infer_input_type on a callable whose annotation
    # resolves to an instance value isn't directly testable via normal hints.
    # Instead, verify via a class with __annotations__ that produces a non-type value.
    class WeirdCallable:
        __annotations__ = {"input": not_a_type_instance}  # type: ignore[assignment]

        def __call__(self, input):
            return str(input)

    assert _infer_input_type(WeirdCallable()) is None


# --- Integration: Interact forwards input_type to InputGenerator ---


class RecordingTrace(Trace[Interaction[str, str]], frozen=True):
    def _repr_prompt_(self) -> str:
        return ""


@InputGenerator.register("recording_generator")
class RecordingGenerator(InputGenerator[RecordingTrace]):
    received_input_type: type[Any] | None = None

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
async def test_interact_passes_str_input_type_for_str_annotated_target():
    gen = RecordingGenerator()

    def target(inputs: str) -> str:
        return inputs

    interact = Interact(inputs=gen, outputs=target)
    trace = RecordingTrace()
    agen = interact.generate(trace)
    await anext(agen)
    assert gen.received_input_type is str


@pytest.mark.asyncio
async def test_interact_forwards_list_input_type_to_generator():
    gen = RecordingGenerator()

    def target(inputs: list[str]) -> str:
        return ", ".join(inputs)

    interact = Interact(inputs=gen, outputs=target)
    trace = RecordingTrace()
    agen = interact.generate(trace)
    await anext(agen)
    assert gen.received_input_type == list[str]
