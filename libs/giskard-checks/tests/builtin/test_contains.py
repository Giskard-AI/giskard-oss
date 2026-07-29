"""Tests for the ContainsAny and ContainsAll checks."""

import pytest
from giskard.checks import CheckStatus, ContainsAll, ContainsAny, Interaction, Trace


async def test_contains_any_success() -> None:
    """Test that ContainsAny passes when at least one value is found."""
    check = ContainsAny(
        text="Hello World Python", values=["world", "java"], case_sensitive=False
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.message is not None
    assert "at least one of the values" in result.message
    assert "world" in result.details["matched_values"]


async def test_contains_any_failure() -> None:
    """Test that ContainsAny fails when no values are found."""
    check = ContainsAny(
        text="Hello World", values=["python", "java"], case_sensitive=False
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "does not contain any of the provided values" in result.message


async def test_contains_all_success() -> None:
    """Test that ContainsAll passes when all values are found."""
    check = ContainsAll(
        text="Hello World Python", values=["world", "python"], case_sensitive=False
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.message is not None
    assert "contains all of the provided values" in result.message
    assert len(result.details["missing_values"]) == 0


async def test_contains_all_failure() -> None:
    """Test that ContainsAll fails when some values are missing."""
    check = ContainsAll(
        text="Hello World Python", values=["world", "java"], case_sensitive=False
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "is missing the following values: java" in result.message
    assert "java" in result.details["missing_values"]


async def test_case_sensitive_matching() -> None:
    """Test case-sensitive matching behavior."""
    # Case-sensitive: should fail
    check_sensitive = ContainsAny(
        text="Hello World", values=["world"], case_sensitive=True
    )
    result = await check_sensitive.run(Trace())
    assert result.status == CheckStatus.FAIL

    # Case-insensitive: should pass
    check_insensitive = ContainsAny(
        text="Hello World", values=["world"], case_sensitive=False
    )
    result = await check_insensitive.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_extraction_from_trace() -> None:
    """Test extracting both text and values from trace."""
    check = ContainsAll(
        text_key="trace.last.outputs.response",
        values_key="trace.last.inputs.topics",
    )
    interaction = Interaction(
        inputs={"topics": ["Paris", "France"]},
        outputs={"response": "The capital of France is Paris."},
    )
    result = await check.run(Trace(interactions=[interaction]))
    assert result.status == CheckStatus.PASS
    assert result.details["text"] == "The capital of France is Paris."
    assert result.details["values"] == ["Paris", "France"]


async def test_invalid_values_type() -> None:
    """Test behavior when values is not a list of strings."""
    check = ContainsAny(
        text="Hello 123",
        values_key="trace.last.inputs.topics",
    )
    interaction = Interaction(
        inputs={"topics": "not a list"},
        outputs={"response": "response"},
    )
    result = await check.run(Trace(interactions=[interaction]))
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "must be a list of strings" in result.message


async def test_unicode_normalization() -> None:
    """Test that Unicode normalization works correctly."""
    # Using NFKC normalization (default)
    check = ContainsAny(
        text="Hello Ａ World",
        values=["A", "B"],
        normalization_form="NFKC",
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_cannot_provide_both_values_and_values_key() -> None:
    """Test that providing both values and values_key raises an error."""
    with pytest.raises(ValueError, match="Exactly one"):
        ContainsAny(
            text="Hello World",
            values=["hello"],
            values_key="trace.last.inputs.topics",
        )
