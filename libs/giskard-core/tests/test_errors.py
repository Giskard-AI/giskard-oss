"""Tests for error handling."""

from giskard.core import Error


def test_error_creation():
    """Test that Error can be created with a message."""
    error = Error(message="Something went wrong")
    assert error.message == "Something went wrong"


def test_error_string_representation():
    """Test that Error has correct string representation."""
    error = Error(message="Something went wrong")
    assert str(error) == "ERROR: Something went wrong"


def test_error_serialization():
    """Test that Error can be serialized."""
    error = Error(message="Something went wrong")
    serialized = error.model_dump()
    assert serialized == {"message": "Something went wrong"}


def test_error_deserialization():
    """Test that Error can be deserialized from dict."""
    data = {"message": "Something went wrong"}
    error = Error.model_validate(data)
    assert error.message == "Something went wrong"


def test_error_json_serialization():
    """Test that Error can be serialized to JSON."""
    error = Error(message="Something went wrong")
    json_str = error.model_dump_json()
    assert '"message":"Something went wrong"' in json_str
