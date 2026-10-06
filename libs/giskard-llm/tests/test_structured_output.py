"""Tests for shared structured-output schema normalization."""

import logging

import pytest
from giskard.llm.errors import BadRequestError
from giskard.llm.structured_output import (
    is_lossy_schema_mutation,
    normalize_pydantic_json_schema,
    object_schema_paths_missing_additional_properties_false,
    reject_user_schema_mutation_param,
    reset_schema_mutation_warnings_for_tests,
)
from pydantic import BaseModel, Field


class NestedInnerModel(BaseModel):
    value: str


class NestedOutputModel(BaseModel):
    inner: NestedInnerModel


class _FlatOutputModel(BaseModel):
    value: int
    count: int


class _ModelWithDefaultField(BaseModel):
    value: str = Field(default="hello")


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


def test_google_flat_model_preserves_properties_and_required():
    schema = normalize_pydantic_json_schema(
        _FlatOutputModel,
        profile="google",
        provider="google",
        mode="ignore",
    )
    assert schema["properties"]["value"]["type"] == "integer"
    assert schema["properties"]["count"]["type"] == "integer"
    assert schema["required"] == ["value", "count"]
    assert schema.get("$defs") in (None, {})
    assert schema["additionalProperties"] is False


def test_google_strip_preserves_property_and_def_names():
    original = NestedOutputModel.model_json_schema()
    normalized = normalize_pydantic_json_schema(
        NestedOutputModel,
        profile="google",
        provider="google",
        mode="ignore",
    )
    assert "inner" in normalized.get("properties", {})
    assert "NestedInnerModel" in normalized.get("$defs", {})
    assert "value" in normalized["$defs"]["NestedInnerModel"].get("properties", {})
    assert is_lossy_schema_mutation(original, normalized, "google") is False


def test_google_strip_removes_unsupported_keywords_is_lossy(caplog):
    original = _ModelWithDefaultField.model_json_schema()
    assert "default" in str(original)
    reset_schema_mutation_warnings_for_tests()
    with caplog.at_level(logging.WARNING):
        normalize_pydantic_json_schema(
            _ModelWithDefaultField,
            profile="google",
            provider="google",
            mode="warn",
        )
    assert any("lossily normalized" in r.message for r in caplog.records)


def test_schema_mutation_raise_blocks_lossy_google_normalization():
    with pytest.raises(BadRequestError, match="lossily normalized"):
        normalize_pydantic_json_schema(
            _ModelWithDefaultField,
            profile="google",
            provider="google",
            mode="raise",
        )


def test_schema_mutation_raise_allows_openai_additional_properties_only():
    normalize_pydantic_json_schema(
        NestedOutputModel,
        profile="openai",
        provider="openai",
        mode="raise",
    )


def test_schema_mutation_raise_allows_flat_models_all_profiles():
    pytest.importorskip("anthropic")
    normalize_pydantic_json_schema(
        _FlatOutputModel,
        profile="openai",
        provider="openai",
        mode="raise",
    )
    normalize_pydantic_json_schema(
        _FlatOutputModel,
        profile="google",
        provider="google",
        mode="raise",
    )
    normalize_pydantic_json_schema(
        _FlatOutputModel,
        profile="anthropic",
        provider="anthropic",
        mode="raise",
    )


def test_reject_user_schema_mutation_param_on_completion(caplog):
    params = {"temperature": 0.0, "schema_mutation": "raise"}
    with caplog.at_level(logging.WARNING):
        reject_user_schema_mutation_param(params)
    assert params == {"temperature": 0.0}
    assert any("configure-time only" in r.message for r in caplog.records)


def test_schema_mutation_warn_only_on_lossy_and_dedupes(caplog):
    reset_schema_mutation_warnings_for_tests()
    with caplog.at_level(logging.WARNING):
        normalize_pydantic_json_schema(
            NestedOutputModel,
            profile="openai",
            provider="openai",
            mode="warn",
        )
        normalize_pydantic_json_schema(
            _ModelWithDefaultField,
            profile="google",
            provider="google",
            mode="warn",
        )
        for _ in range(6):
            normalize_pydantic_json_schema(
                _ModelWithDefaultField,
                profile="google",
                provider="google",
                mode="warn",
            )
    lossy = [r for r in caplog.records if "lossily normalized" in r.message]
    assert len(lossy) == 1


def test_schema_mutation_ignore_is_silent(caplog):
    reset_schema_mutation_warnings_for_tests()
    with caplog.at_level(logging.WARNING):
        normalize_pydantic_json_schema(
            _ModelWithDefaultField,
            profile="google",
            provider="google",
            mode="ignore",
        )
    assert caplog.records == []
