from collections.abc import AsyncGenerator
from typing import Any, override

import pytest
from giskard.checks import Trace
from giskard.checks.core.input_generator import InputGenerator
from pydantic import BaseModel


class ConcreteTrace(Trace[str, str], frozen=True):
    def _repr_prompt_(self) -> str:
        return ""


class MyModel(BaseModel):
    content: str


@InputGenerator.register("test_typed_generator")
class TypedGenerator(InputGenerator[ConcreteTrace]):
    received_input_type: type | None = None

    @override
    async def __call__(
        self, trace: ConcreteTrace, input_type: type[str | BaseModel] | None = None
    ) -> AsyncGenerator[Any, ConcreteTrace]:
        self.received_input_type = input_type

        if input_type is MyModel:
            yield MyModel(content="hello")
        else:
            assert input_type is str or input_type is None
            yield "hello"


@pytest.mark.asyncio
async def test_input_generator_forwards_input_type():
    gen = TypedGenerator()
    trace = ConcreteTrace()
    agen = gen(trace, input_type=MyModel)
    await anext(agen)
    assert gen.received_input_type is MyModel


@pytest.mark.asyncio
async def test_input_generator_defaults_input_type_to_none():
    gen = TypedGenerator()
    trace = ConcreteTrace()
    agen = gen(trace)
    await anext(agen)
    assert gen.received_input_type is None
