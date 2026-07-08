"""Tests for the JsonValid check."""

from typing import Any

import pytest
from giskard.checks import Check, CheckStatus, Interaction, JsonValid, Trace
from giskard.checks.core.extraction import NoMatch
from pydantic import ValidationError


@pytest.mark.parametrize(
    ("outputs", "expected_parsed"),
    [
        ('{"name": "Alice", "age": 30}', {"name": "Alice", "age": 30}),
        ("null", None),
        ("true", True),
        ("42", 42),
    ],
)
async def test_valid_json_string_passes(outputs: Any, expected_parsed: Any) -> None:
    check = JsonValid()
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs=outputs)
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["parsed_value"] == expected_parsed


@pytest.mark.parametrize(
    "outputs",
    [
        {"name": "Alice", "age": 30},
        [{"id": 1}, {"id": 2}],
        None,
        42,
        "value",
    ],
)
async def test_already_parsed_value_passes(outputs: Any) -> None:
    check = JsonValid(parse=False)
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs=outputs)
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["parsed_value"] == outputs


@pytest.mark.parametrize(
    "outputs",
    [
        '{"name": "Alice"',
        "",
    ],
)
async def test_invalid_json_string_fails(outputs: str) -> None:
    check = JsonValid()
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs=outputs)
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.message is not None
    assert "not valid JSON" in result.message
    assert "error" in result.details


async def test_nested_jsonpath_extraction() -> None:
    check = JsonValid(key="trace.last.outputs.response")
    trace = await Trace.from_interactions(
        Interaction(
            inputs="Return JSON",
            outputs={"response": '{"items": [{"id": 1}, {"id": 2}]}'},
        )
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["parsed_value"] == {"items": [{"id": 1}, {"id": 2}]}


@pytest.mark.parametrize(
    ("schema", "outputs", "parse", "expected_parsed"),
    [
        (
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
            '{"name": "Alice", "age": 30}',
            True,
            {"name": "Alice", "age": 30},
        ),
        ({"type": "integer"}, 42, False, 42),
        ({"type": "string", "minLength": 3, "maxLength": 7}, '"hello"', True, "hello"),
    ],
)
async def test_schema_validation_passes(
    schema: dict[str, Any], outputs: Any, parse: bool, expected_parsed: Any
) -> None:
    check = JsonValid(schema=schema, parse=parse)
    trace = await Trace.from_interactions(
        Interaction(inputs="Return user data", outputs=outputs)
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["parsed_value"] == expected_parsed


@pytest.mark.parametrize(
    ("schema", "outputs", "parse", "expected_error"),
    [
        (
            {
                "type": "object",
                "properties": {"age": {"type": "integer"}},
                "required": ["age"],
            },
            '{"age": "old"}',
            True,
            "'old' is not of type 'integer'",
        ),
        ({"type": "string"}, 42, False, "42 is not of type 'string'"),
    ],
)
async def test_schema_validation_fails(
    schema: dict[str, Any], outputs: Any, parse: bool, expected_error: str
) -> None:
    check = JsonValid(schema=schema, parse=parse)
    trace = await Trace.from_interactions(
        Interaction(inputs="Return user data", outputs=outputs)
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.message is not None
    assert "does not match the provided schema" in result.message
    assert result.details["error"] == expected_error


def test_invalid_schema_fails_at_instantiation() -> None:
    with pytest.raises(ValidationError) as exc_info:
        JsonValid(schema={"type": "not-a-json-schema-type"})

    assert "Provided JSON Schema is invalid" in str(exc_info.value)


async def test_unresolvable_schema_ref_returns_error() -> None:
    check = JsonValid(schema={"$ref": "https://unreachable.example.com/schema"})
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs='{"name": "Alice"}')
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert result.message is not None
    assert "unresolvable $ref" in result.message
    assert "https://unreachable.example.com/schema" in result.message


async def test_missing_key_fails() -> None:
    check = JsonValid(key="trace.last.outputs.missing")
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs={"response": "{}"})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert isinstance(result.details["value"], NoMatch)
    assert result.message == "No value found for key 'trace.last.outputs.missing'."


async def test_non_serializable_value_fails() -> None:
    check = JsonValid(parse=False)
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs={"values": {1, 2, 3}})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.message is not None
    assert "trace.last.outputs" in result.message
    assert "not JSON serializable" in result.message
    assert "error" in result.details


def test_json_valid_is_exported() -> None:
    assert JsonValid.__name__ == "JsonValid"


def test_json_valid_serialization_roundtrip() -> None:
    check = JsonValid(
        key="trace.last.outputs.response", schema={"type": "object"}, parse=False
    )

    data = check.model_dump()
    restored = Check.model_validate(data)

    assert data["kind"] == "json_valid"
    assert data["schema"] == {"type": "object"}
    assert data["parse"] is False
    assert isinstance(restored, JsonValid)
    assert restored.key == "trace.last.outputs.response"
    assert restored.expected_schema == {"type": "object"}
    assert restored.parse is False


async def test_parse_false_accepts_plain_string() -> None:
    check = JsonValid(parse=False)
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs="value")
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["parsed_value"] == "value"


async def test_parse_true_rejects_plain_string() -> None:
    check = JsonValid(parse=True)
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs="value")
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "not valid JSON" in result.message


async def test_parse_true_rejects_non_string() -> None:
    check = JsonValid(parse=True)
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs={"name": "Alice"})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "is not a string" in result.message


async def test_parse_false_schema_match_on_string() -> None:
    check = JsonValid(parse=False, schema={"type": "string"})
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs="value")
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS


async def test_parse_false_schema_mismatch_on_string() -> None:
    check = JsonValid(parse=False, schema={"type": "number"})
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs="value")
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "does not match the provided schema" in result.message
