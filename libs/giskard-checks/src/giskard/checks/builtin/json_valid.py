import asyncio
import pytest

from giskard.checks import JsonValid
from giskard.core.core import Trace


def test_valid_json():
    trace = Trace(outputs={"result": '{"name": "John"}'})

    check = JsonValid(key="result")
    result = asyncio.run(check.run(trace))

    assert result.passed
    assert "Valid JSON" in result.message


def test_invalid_json():
    trace = Trace(outputs={"result": '{"name": John}'})  # invalid JSON

    check = JsonValid(key="result")
    result = asyncio.run(check.run(trace))

    assert not result.passed
    assert "Invalid JSON" in result.message


def test_schema_validation_success():
    trace = Trace(outputs={"result": '{"age": 25}'})

    schema = {
        "type": "object",
        "properties": {
            "age": {"type": "number"}
        },
        "required": ["age"]
    }

    check = JsonValid(key="result", expected_schema=schema)
    result = asyncio.run(check.run(trace))

    assert result.passed


def test_schema_validation_failure():
    trace = Trace(outputs={"result": '{"age": "twenty"}'})

    schema = {
        "type": "object",
        "properties": {
            "age": {"type": "number"}
        },
        "required": ["age"]
    }

    check = JsonValid(key="result", expected_schema=schema)
    result = asyncio.run(check.run(trace))

    assert not result.passed
    assert "Schema validation failed" in result.message


def test_key_not_found():
    trace = Trace(outputs={"data": '{"name": "John"}'})

    check = JsonValid(key="missing")
    result = asyncio.run(check.run(trace))

    assert not result.passed
    assert "Key not found" in result.message