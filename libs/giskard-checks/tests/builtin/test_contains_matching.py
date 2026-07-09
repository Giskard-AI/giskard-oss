"""Tests for the ContainsAny and ContainsAll checks."""

import pytest
from giskard.checks import CheckStatus, ContainsAll, ContainsAny, Interaction, Trace
from giskard.checks.core.extraction import NoMatch


async def test_contains_any_passes_with_one_match() -> None:
    """ContainsAny passes when at least one value is found."""
    check = ContainsAny(
        text="The answer covers pricing and support.",
        values=["pricing", "refunds", "shipping"],
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["matched"] == ["pricing"]
    assert result.details["missing"] == ["refunds", "shipping"]


async def test_contains_any_passes_with_all_matches() -> None:
    """ContainsAny passes when every value is found."""
    check = ContainsAny(
        text="pricing, refunds and shipping",
        values=["pricing", "refunds", "shipping"],
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["matched"] == ["pricing", "refunds", "shipping"]


async def test_contains_any_fails_with_no_matches() -> None:
    """ContainsAny fails when none of the values are found."""
    check = ContainsAny(
        text="The answer is unrelated.",
        values=["pricing", "refunds", "shipping"],
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "does not contain any" in result.message
    assert result.details["matched"] == []
    assert result.details["missing"] == ["pricing", "refunds", "shipping"]


async def test_contains_all_passes_when_all_present() -> None:
    """ContainsAll passes when every value is found."""
    check = ContainsAll(
        text="The answer covers pricing, refunds and shipping.",
        values=["pricing", "refunds", "shipping"],
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.message is not None
    assert "contains all" in result.message
    assert result.details["missing"] == []


async def test_contains_all_fails_when_partial_match() -> None:
    """ContainsAll fails when only some values are found."""
    check = ContainsAll(
        text="The answer covers pricing only.",
        values=["pricing", "refunds", "shipping"],
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "missing" in result.message
    assert result.details["matched"] == ["pricing"]
    assert result.details["missing"] == ["refunds", "shipping"]


async def test_case_sensitivity() -> None:
    """Case sensitivity is configurable for both checks."""
    check_sensitive = ContainsAny(
        text="Hello World", values=["world"], case_sensitive=True
    )
    result = await check_sensitive.run(Trace())
    assert result.status == CheckStatus.FAIL

    check_insensitive = ContainsAny(
        text="Hello World", values=["world"], case_sensitive=False
    )
    result = await check_insensitive.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_text_and_values_from_trace() -> None:
    """Both text and values can be extracted from the trace via JSONPath."""
    check = ContainsAll(
        text_key="trace.last.outputs.response",
        values_key="trace.last.inputs.expected_topics",
    )
    interaction = Interaction(
        inputs={"expected_topics": ["Paris", "Eiffel"]},
        outputs={"response": "The Eiffel Tower is in Paris."},
    )
    result = await check.run(Trace(interactions=[interaction]))
    assert result.status == CheckStatus.PASS


async def test_unicode_normalization() -> None:
    """Unicode normalization is applied before matching."""
    check = ContainsAny(
        text="Hello Ａ World",
        values=["A"],
        normalization_form="NFKC",
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_empty_values_list_fails() -> None:
    """An empty values list is a validation failure, not a pass or crash."""
    check = ContainsAny(text="Hello", values=[])
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "empty" in result.message


async def test_missing_values_in_trace() -> None:
    """Missing values in the trace produce a failure result, not an exception."""
    check = ContainsAny(
        text="Some text",
        values_key="trace.last.inputs.nonexistent",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert (
        "No value found for values key 'trace.last.inputs.nonexistent'."
        in result.message
    )
    assert isinstance(result.details["values"], NoMatch)


async def test_missing_text_in_trace() -> None:
    """Missing text in the trace produces a failure result, not an exception."""
    check = ContainsAny(
        text_key="trace.last.outputs.nonexistent",
        values=["test"],
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert (
        "No value found for text key 'trace.last.outputs.nonexistent'" in result.message
    )
    assert isinstance(result.details["text"], NoMatch)


async def test_non_list_values_fails() -> None:
    """A non-list value for `values` is a validation failure, not a crash."""
    check = ContainsAny(
        text_key="trace.last.outputs.response",
        values_key="trace.last.outputs.response",
    )
    interaction = Interaction(inputs={}, outputs={"response": "not a list"})
    result = await check.run(Trace(interactions=[interaction]))
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "not a list of strings" in result.message


async def test_cannot_provide_both_values_and_values_key() -> None:
    """Providing both values and values_key raises at construction time."""
    with pytest.raises(ValueError, match="Exactly one"):
        ContainsAny(
            text="Hello World",
            values=["hello"],
            values_key="trace.last.inputs.key",
        )


async def test_cannot_provide_neither_values_nor_values_key() -> None:
    """Providing neither values nor values_key raises at construction time."""
    with pytest.raises(ValueError, match="Exactly one"):
        ContainsAny(text="Hello World")
