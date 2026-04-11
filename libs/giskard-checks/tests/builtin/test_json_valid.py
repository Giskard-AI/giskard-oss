import sys
import os
import pytest

sys.path.append(
    os.path.abspath("libs/giskard-checks/src")
)

from giskard.checks import JsonValid


class DummyTrace:
    def __init__(self, output):
        self.outputs = output


@pytest.mark.asyncio
async def test_valid_json():
    check = JsonValid()
    trace = DummyTrace('{"name": "Alice"}')

    result = await check.run(trace)
    assert result.passed


@pytest.mark.asyncio
async def test_invalid_json():
    check = JsonValid()
    trace = DummyTrace('{"name": Alice}')

    result = await check.run(trace)
    assert not result.passed


@pytest.mark.asyncio
async def test_schema_valid():
    schema = {
        "type": "object",
        "properties": {"age": {"type": "number"}},
        "required": ["age"]
    }

    check = JsonValid(schema=schema)
    trace = DummyTrace('{"age": 25}')

    result = await check.run(trace)
    assert result.passed


@pytest.mark.asyncio
async def test_schema_invalid():
    schema = {
        "type": "object",
        "properties": {"age": {"type": "number"}},
        "required": ["age"]
    }

    check = JsonValid(schema=schema)
    trace = DummyTrace('{"age": "twenty"}')

    result = await check.run(trace)
    assert not result.passed