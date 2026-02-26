"""Tests for the StringMatching check."""

import pytest
from giskard.checks import CheckStatus, Interaction, StringMatching, Trace
from giskard.checks.core.extraction import NoMatch


async def test_run_returns_success_with_direct_values() -> None:
    """Test that check passes when keyword is found in text with direct values."""
    check = StringMatching(text="Hello World", keyword="world", case_sensitive=False)
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.message is not None
    assert "contains the keyword" in result.message
    assert result.details["text"] == "Hello World"
    assert result.details["keyword"] == "world"


async def test_run_returns_failure_when_keyword_not_found() -> None:
    """Test that check fails when keyword is not found in text."""
    check = StringMatching(text="Hello World", keyword="Python", case_sensitive=False)
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "does not contain the keyword" in result.message
    assert result.details["text"] == "Hello World"
    assert result.details["keyword"] == "Python"


async def test_case_sensitive_matching() -> None:
    """Test case-sensitive matching behavior."""
    # Case-sensitive: should fail
    check_sensitive = StringMatching(
        text="Hello World", keyword="world", case_sensitive=True
    )
    result = await check_sensitive.run(Trace())
    assert result.status == CheckStatus.FAIL

    # Case-insensitive: should pass
    check_insensitive = StringMatching(
        text="Hello World", keyword="world", case_sensitive=False
    )
    result = await check_insensitive.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_text_and_keyword_from_trace() -> None:
    """Test extracting both text and keyword from trace."""
    check = StringMatching(
        text_key="trace.last.outputs.response",
        keyword_key="trace.last.inputs.expected",
    )
    interaction = Interaction(
        inputs={"expected": "Paris"},
        outputs={"response": "The capital of France is Paris."},
    )
    result = await check.run(Trace(interactions=[interaction]))
    assert result.status == CheckStatus.PASS
    assert result.details["text"] == "The capital of France is Paris."
    assert result.details["keyword"] == "Paris"


async def test_text_from_trace_keyword_direct() -> None:
    """Test extracting text from trace with direct keyword."""
    check = StringMatching(
        text_key="trace.last.outputs.answer",
        keyword="Python",
        case_sensitive=False,
    )
    interaction = Interaction(
        inputs={"query": "What is Python?"},
        outputs={"answer": "Python is a programming language."},
    )
    result = await check.run(Trace(interactions=[interaction]))
    assert result.status == CheckStatus.PASS
    assert result.details["text"] == "Python is a programming language."
    assert result.details["keyword"] == "Python"


async def test_keyword_from_trace_text_direct() -> None:
    """Test extracting keyword from trace with direct text."""
    check = StringMatching(
        text="The Eiffel Tower is in Paris.",
        keyword_key="trace.last.inputs.city",
    )
    interaction = Interaction(
        inputs={"city": "Paris"},
        outputs={"response": "Confirmed"},
    )
    result = await check.run(Trace(interactions=[interaction]))
    assert result.status == CheckStatus.PASS
    assert result.details["text"] == "The Eiffel Tower is in Paris."
    assert result.details["keyword"] == "Paris"


async def test_unicode_normalization() -> None:
    """Test that Unicode normalization works correctly."""
    # Using NFKC normalization (default) - NFKC normalizes compatibility characters
    # like full-width characters and superscripts, but not accents
    # Example: full-width A (Ａ) normalizes to regular A
    check = StringMatching(
        text="Hello Ａ World",
        keyword="A",
        normalization_form="NFKC",
        case_sensitive=False,
    )
    result = await check.run(Trace())
    # NFKC should normalize full-width Ａ to regular A, so it should match
    assert result.status == CheckStatus.PASS


async def test_unicode_normalization_superscript() -> None:
    """Test that Unicode normalization works with superscripts."""
    # NFKC normalizes superscripts (²) to regular characters (2)
    check = StringMatching(
        text="x² + y² = z²",
        keyword="2",
        normalization_form="NFKC",
        case_sensitive=False,
    )
    result = await check.run(Trace())
    # NFKC should normalize ² to 2, so it should match
    assert result.status == CheckStatus.PASS


async def test_whitespace_normalization() -> None:
    """Test that whitespace is normalized (multiple spaces collapsed)."""
    check = StringMatching(
        text="Hello    World   Test",
        keyword="World Test",
        case_sensitive=False,
    )
    result = await check.run(Trace())
    # Should match despite different spacing
    assert result.status == CheckStatus.PASS


async def test_whitespace_trimming() -> None:
    """Test that leading and trailing whitespace is trimmed."""
    check = StringMatching(
        text="  Hello World  ",
        keyword="Hello World",
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_missing_keyword_in_trace() -> None:
    """Test behavior when keyword cannot be extracted from trace."""
    check = StringMatching(
        text="Some text",
        keyword_key="trace.last.inputs.nonexistent",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert (
        "No value found for keyword key 'trace.last.inputs.nonexistent'."
        in result.message
    )
    assert isinstance(result.details["keyword"], NoMatch)
    assert result.details["keyword"].key == "trace.last.inputs.nonexistent"


async def test_missing_text_in_trace() -> None:
    """Test behavior when text cannot be extracted from trace."""
    check = StringMatching(
        text_key="trace.last.outputs.nonexistent",
        keyword="test",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert (
        "No value found for text key 'trace.last.outputs.nonexistent'" in result.message
    )
    assert isinstance(result.details["text"], NoMatch)
    assert result.details["text"].key == "trace.last.outputs.nonexistent"


async def test_default_text_key() -> None:
    """Test that default text_key (trace.last.outputs) works correctly."""
    check = StringMatching(keyword="response")
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs="This is a response",
    )
    result = await check.run(Trace(interactions=[interaction]))
    # Default text_key extracts trace.last.outputs which is a dict
    # The dict string representation should contain "response"
    assert result.status == CheckStatus.PASS


async def test_empty_trace_with_direct_values() -> None:
    """Test that check works with direct values even with empty trace."""
    check = StringMatching(text="Hello", keyword="Hello")
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_trace_last_property() -> None:
    """Test using trace.last property for extraction."""
    check = StringMatching(
        text_key="trace.last.outputs.message",
        keyword="Alice",
    )
    interaction1 = Interaction(
        inputs={"name": "Bob"},
        outputs={"message": "Hello Bob"},
    )
    interaction2 = Interaction(
        inputs={"name": "Alice"},
        outputs={"message": "Hello Alice"},
    )
    trace = Trace(interactions=[interaction1, interaction2])
    result = await check.run(trace)
    # Should extract from last interaction
    assert result.status == CheckStatus.PASS
    assert result.details["text"] == "Hello Alice"


async def test_multiple_interactions_uses_last() -> None:
    """Test that check uses the last interaction when multiple exist."""
    check = StringMatching(
        text_key="trace.last.outputs.response",
        keyword="Second",
    )
    interaction1 = Interaction(
        inputs={"query": "First"},
        outputs={"response": "First response"},
    )
    interaction2 = Interaction(
        inputs={"query": "Second"},
        outputs={"response": "Second response"},
    )
    trace = Trace(interactions=[interaction1, interaction2])
    result = await check.run(trace)
    # Should use last interaction
    assert result.status == CheckStatus.PASS
    assert "Second response" in str(result.details["text"])


async def test_normalization_form_none() -> None:
    """Test that no normalization is applied when normalization_form is None."""
    check = StringMatching(
        text="café",
        keyword="cafe",
        normalization_form=None,
        case_sensitive=False,
    )
    result = await check.run(Trace())
    # Without normalization, é and e should not match
    assert result.status == CheckStatus.FAIL


async def test_different_normalization_forms() -> None:
    """Test different Unicode normalization forms."""
    # Test NFC
    check_nfc = StringMatching(
        text="café",
        keyword="café",
        normalization_form="NFC",
        case_sensitive=False,
    )
    result = await check_nfc.run(Trace())
    assert result.status == CheckStatus.PASS

    # Test NFD
    check_nfd = StringMatching(
        text="café",
        keyword="café",
        normalization_form="NFD",
        case_sensitive=False,
    )
    result = await check_nfd.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_keyword_at_start_of_text() -> None:
    """Test matching when keyword appears at the start of text."""
    check = StringMatching(
        text="Python is great",
        keyword="Python",
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_keyword_at_end_of_text() -> None:
    """Test matching when keyword appears at the end of text."""
    check = StringMatching(
        text="The capital is Paris",
        keyword="Paris",
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_keyword_in_middle_of_text() -> None:
    """Test matching when keyword appears in the middle of text."""
    check = StringMatching(
        text="Hello World Test",
        keyword="World",
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_partial_word_match() -> None:
    """Test that partial word matches work (substring matching)."""
    check = StringMatching(
        text="Python programming",
        keyword="thon",
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_empty_text() -> None:
    """Test behavior with empty text."""
    check = StringMatching(text="", keyword="test")
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL


async def test_empty_keyword() -> None:
    """Test behavior with empty keyword."""
    check = StringMatching(text="Hello", keyword="")
    result = await check.run(Trace())
    # Empty string should be found in any text
    assert result.status == CheckStatus.PASS


async def test_unicode_e_acute_nfc_nfd_matching() -> None:
    """Test that 'é' (U+00E9) matches 'é' (U+0065 U+0301) with NFC normalization."""
    # U+00E9 is the composed form (NFC)
    # U+0065 U+0301 is the decomposed form (NFD): 'e' + combining acute accent
    text_nfc = "café"  # Uses U+00E9
    keyword_nfd = "caf\u0065\u0301"  # Uses U+0065 U+0301

    # With NFC normalization, both should normalize to the same form
    check = StringMatching(
        text=text_nfc,
        keyword=keyword_nfd,
        normalization_form="NFC",
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_unicode_e_acute_nfd_nfc_matching() -> None:
    """Test that 'é' (U+0065 U+0301) matches 'é' (U+00E9) with NFD normalization."""
    # U+0065 U+0301 is the decomposed form (NFD)
    # U+00E9 is the composed form (NFC)
    text_nfd = "caf\u0065\u0301"  # Uses U+0065 U+0301
    keyword_nfc = "café"  # Uses U+00E9

    # With NFD normalization, both should normalize to the same form
    check = StringMatching(
        text=text_nfd,
        keyword=keyword_nfc,
        normalization_form="NFD",
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_unicode_e_acute_nfkc_matching() -> None:
    """Test that 'é' in different forms matches with NFKC normalization."""
    # NFKC should also normalize both forms to the same representation
    text_nfc = "café"  # Uses U+00E9
    keyword_nfd = "caf\u0065\u0301"  # Uses U+0065 U+0301

    check = StringMatching(
        text=text_nfc,
        keyword=keyword_nfd,
        normalization_form="NFKC",
        case_sensitive=False,
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_cannot_provide_both_keyword_and_keyword_key() -> None:
    """Test that providing both keyword and keyword_key raises an error."""
    with pytest.raises(ValueError, match="Exactly one"):
        StringMatching(
            text="Hello World",
            keyword="hello",
            keyword_key="trace.last.inputs.key",
        )


async def test_unicode_e_acute_no_normalization_fails() -> None:
    """Test that 'é' in different forms does NOT match without normalization."""
    # Without normalization, different Unicode representations should not match
    text_nfc = "café"  # Uses U+00E9
    keyword_nfd = "caf\u0065\u0301"  # Uses U+0065 U+0301

    check = StringMatching(
        text=text_nfc,
        keyword=keyword_nfd,
        normalization_form=None,
        case_sensitive=False,
    )
    result = await check.run(Trace())
    # Without normalization, they should not match
    assert result.status == CheckStatus.FAIL
