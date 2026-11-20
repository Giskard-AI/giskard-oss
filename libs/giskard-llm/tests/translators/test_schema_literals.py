"""Schema closure preserves instance values in structured-output requests."""

from copy import deepcopy
from typing import Any, cast

import pytest
from giskard.llm.translators.openai_chat import OpenAIChatTranslator
from giskard.llm.translators.openai_response import OpenAIResponseTranslator
from giskard.llm.types import UserMessage
from giskard.llm.types._serialization import close_object_schemas
from pydantic import BaseModel

from .sdk_payload_validation import (
    validate_openai_completion_params,
    validate_openai_response_params,
)


@pytest.mark.parametrize("operation", ["chat", "response"])
def test_pydantic_default_is_preserved_in_structured_output(operation: str):
    class Output(BaseModel):
        metadata: dict[str, str] = {"type": "object"}

    expected = Output.model_json_schema()["properties"]["metadata"]["default"]
    if operation == "chat":
        payload = OpenAIChatTranslator.to_openai(
            "test-model", [UserMessage(content="Hello")], response_format=Output
        )
        validate_openai_completion_params(payload)
        schema = cast(dict[str, Any], cast(object, payload))["response_format"][
            "json_schema"
        ]["schema"]
    else:
        payload = OpenAIResponseTranslator.to_openai(
            "test-model", "Hello", response_format=Output
        )
        validate_openai_response_params(payload)
        schema = cast(dict[str, Any], cast(object, payload))["text"]["format"]["schema"]

    assert schema["properties"]["metadata"]["default"] == expected
    assert schema["properties"]["metadata"]["additionalProperties"] == {
        "type": "string"
    }
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("keyword", ["const", "enum", "default", "examples"])
def test_responses_literal_values_are_not_rewritten(keyword: str):
    value = {"type": "object", "value": 1}
    literal = [value] if keyword in ("enum", "examples") else value
    schema = {
        "type": "object",
        "properties": {"type": {"type": "string"}, "value": {"type": "integer"}},
        "required": ["type", "value"],
        keyword: literal,
    }
    response_format = {
        "type": "json_schema",
        "name": "LiteralOutput",
        "schema": schema,
    }
    before = deepcopy(response_format)
    payload = OpenAIResponseTranslator.to_openai(
        "test-model", "Hello", response_format=response_format
    )
    validate_openai_response_params(payload)
    output_schema = cast(dict[str, Any], cast(object, payload))["text"]["format"][
        "schema"
    ]

    assert output_schema[keyword] == literal
    assert output_schema["additionalProperties"] is False
    assert response_format == before


@pytest.mark.parametrize("name", ["default", "enum"])
def test_properties_named_like_literal_keywords_are_still_schemas(name: str):
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {name: {"type": "object", "properties": {}}},
    }

    assert close_object_schemas(schema) is schema
    assert schema["additionalProperties"] is False
    assert schema["properties"][name]["additionalProperties"] is False


@pytest.mark.parametrize(
    "keyword",
    ["$defs", "definitions", "patternProperties", "dependentSchemas", "dependencies"],
)
def test_schema_maps_close_their_object_values(keyword: str):
    child = {"type": "object", "properties": {}}
    schema = {keyword: {"entry": child}}

    close_object_schemas(schema)

    assert child["additionalProperties"] is False


@pytest.mark.parametrize(
    "keyword",
    ["allOf", "anyOf", "oneOf", "prefixItems", "items"],
)
def test_schema_arrays_close_their_object_values(keyword: str):
    child = {"type": "object", "properties": {}}
    schema = {keyword: [child]}

    close_object_schemas(schema)

    assert child["additionalProperties"] is False


def test_map_schemas_remain_open_and_unknown_annotations_are_preserved():
    value_schema = {"type": "object", "properties": {}}
    annotation = {"type": "object"}
    schema = {
        "type": "object",
        "additionalProperties": value_schema,
        "x-example": annotation,
    }

    close_object_schemas(schema)

    assert schema["additionalProperties"] is value_schema
    assert value_schema["additionalProperties"] is False
    assert annotation == {"type": "object"}
