"""Unit tests for JSONPath validation in extraction.py."""

import pytest
from giskard.checks.core.extraction import JsonPathStr, _validate_jsonpath_syntax
from pydantic import BaseModel, ValidationError


class TestValidateJsonpathSyntax:
    """Tests for the _validate_jsonpath_syntax validator function."""

    def test_valid_simple_path(self):
        assert _validate_jsonpath_syntax("trace.last.outputs") == "trace.last.outputs"

    def test_valid_index_path(self):
        assert (
            _validate_jsonpath_syntax("trace.interactions[-1].outputs")
            == "trace.interactions[-1].outputs"
        )

    def test_valid_wildcard_path(self):
        assert (
            _validate_jsonpath_syntax("trace.interactions[*].outputs")
            == "trace.interactions[*].outputs"
        )

    def test_valid_nested_path(self):
        assert (
            _validate_jsonpath_syntax("trace.last.metadata.context")
            == "trace.last.metadata.context"
        )

    def test_valid_metadata_reference(self):
        assert (
            _validate_jsonpath_syntax("trace.last.metadata.reference_text")
            == "trace.last.metadata.reference_text"
        )

    def test_syntax_error_unclosed_bracket(self):
        with pytest.raises(ValueError, match="Invalid JSONPath expression"):
            _validate_jsonpath_syntax("trace.last.outputs[")

    def test_syntax_error_double_bracket(self):
        with pytest.raises(ValueError, match="Invalid JSONPath expression"):
            _validate_jsonpath_syntax("trace.last.outputs[[")

    def test_missing_trace_prefix(self):
        with pytest.raises(ValueError, match="path must start with 'trace\\.'"):
            _validate_jsonpath_syntax("last.outputs")

    def test_root_typo(self):
        with pytest.raises(ValueError, match="path must start with 'trace\\.'"):
            _validate_jsonpath_syntax("tras.last.outputs")

    def test_empty_string_missing_prefix(self):
        with pytest.raises(ValueError, match="path must start with 'trace\\.'"):
            _validate_jsonpath_syntax("")


class TestJsonPathStrAnnotatedType:
    """Tests for JsonPathStr as a Pydantic Annotated field type."""

    def test_valid_path_accepted(self):
        class Model(BaseModel):
            key: JsonPathStr

        m = Model(key="trace.last.outputs")
        assert m.key == "trace.last.outputs"

    def test_invalid_syntax_raises_validation_error(self):
        class Model(BaseModel):
            key: JsonPathStr

        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            Model(key="trace.last.outputs[")

    def test_missing_trace_prefix_raises_validation_error(self):
        class Model(BaseModel):
            key: JsonPathStr

        with pytest.raises(ValidationError, match="path must start with 'trace\\.'"):
            Model(key="last.outputs")

    def test_optional_jsonpath_str_accepts_none(self):
        class Model(BaseModel):
            key: JsonPathStr | None = None

        m = Model(key=None)
        assert m.key is None

    def test_optional_jsonpath_str_validates_when_str(self):
        class Model(BaseModel):
            key: JsonPathStr | None = None

        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            Model(key="trace.last.outputs[")
