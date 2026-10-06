"""Tests for shared structured-output schema normalization."""

import logging

import pytest
from giskard.llm.errors import BadRequestError
from giskard.llm.structured_output import (
    normalize_pydantic_json_schema,
    object_schema_paths_missing_additional_properties_false,
)
from pydantic import BaseModel


class NestedInnerModel(BaseModel):
    value: str


class NestedOutputModel(BaseModel):
    inner: NestedInnerModel


class _FlatModel(BaseModel):
    value: int


def test_normalize_openai_nested_sets_additional_properties_on_defs():
    schema = normalize_pydantic_json_schema(
        NestedOutputModel,
        profile="openai",
        provider="openai",
        mode="ignore",
    )
    assert object_schema_paths_missing_additional_properties_false(schema) == []


@pytest.mark.anthropic
def test_normalize_anthropic_uses_sdk_transform():
    pytest.importorskip("anthropic")
    schema = normalize_pydantic_json_schema(
        NestedOutputModel,
        profile="anthropic",
        provider="anthropic",
        mode="ignore",
    )
    assert schema["additionalProperties"] is False
    assert object_schema_paths_missing_additional_properties_false(schema) == []


def test_schema_mutation_raise_blocks_nested_normalization():
    with pytest.raises(BadRequestError, match="normalized"):
        normalize_pydantic_json_schema(
            NestedOutputModel,
            profile="openai",
            provider="openai",
            mode="raise",
        )


def test_schema_mutation_warn_emits_log(caplog):
    with caplog.at_level(logging.WARNING):
        normalize_pydantic_json_schema(
            NestedOutputModel,
            profile="openai",
            provider="openai",
            mode="warn",
        )
    assert any("normalized" in record.message for record in caplog.records)


def test_schema_mutation_ignore_is_silent(caplog):
    with caplog.at_level(logging.WARNING):
        normalize_pydantic_json_schema(
            NestedOutputModel,
            profile="openai",
            provider="openai",
            mode="ignore",
        )
    assert caplog.records == []


def test_openai_profile_always_mutates_pydantic_schema(caplog):
    with caplog.at_level(logging.WARNING):
        normalize_pydantic_json_schema(
            _FlatModel,
            profile="openai",
            provider="openai",
            mode="warn",
        )
    assert any("normalized" in record.message for record in caplog.records)
