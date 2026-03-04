"""Unit tests for JSONPath validation in extraction.py."""

import pytest
from giskard.checks.core.extraction import JSONPathStr, _validate_jsonpath_syntax
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

    def test_just_prefix_is_invalid(self):
        """JSONPath 'trace.' alone should fail parsing as it's incomplete."""
        with pytest.raises(ValueError, match="Invalid JSONPath expression"):
            _validate_jsonpath_syntax("trace.")

    def test_prefix_is_case_sensitive(self):
        """Prefix must be lowercase 'trace.', not 'Trace.' or 'TRACE.'"""
        with pytest.raises(ValueError, match="path must start with 'trace\\.'"):
            _validate_jsonpath_syntax("Trace.last.outputs")

    def test_whitespace_prefix_is_invalid(self):
        """Leading whitespace should be rejected."""
        with pytest.raises(ValueError, match="path must start with 'trace\\.'"):
            _validate_jsonpath_syntax(" trace.last.outputs")

    def test_valid_descendants_operator(self):
        """JSONPath descendants operator (..) should be valid."""
        assert _validate_jsonpath_syntax("trace..outputs") == "trace..outputs"


class TestJSONPathStrAnnotatedType:
    """Tests for JSONPathStr as a Pydantic Annotated field type."""

    def test_valid_path_accepted(self):
        class Model(BaseModel):
            key: JSONPathStr

        m = Model(key="trace.last.outputs")
        assert m.key == "trace.last.outputs"

    def test_invalid_syntax_raises_validation_error(self):
        class Model(BaseModel):
            key: JSONPathStr

        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            Model(key="trace.last.outputs[")

    def test_missing_trace_prefix_raises_validation_error(self):
        class Model(BaseModel):
            key: JSONPathStr

        with pytest.raises(ValidationError, match="path must start with 'trace\\.'"):
            Model(key="last.outputs")

    def test_optional_jsonpath_str_accepts_none(self):
        class Model(BaseModel):
            key: JSONPathStr | None = None

        m = Model(key=None)
        assert m.key is None

    def test_optional_jsonpath_str_validates_when_provided(self):
        """Optional fields should still validate when a string value is provided."""

        class Model(BaseModel):
            key: JSONPathStr | None

        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            Model(key="trace.last.outputs[")
