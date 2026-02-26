"""Tests for RegexMatching check."""

import pytest
from giskard.checks import CheckStatus, Interaction, RegexMatching, Trace


# Basic regex patterns
async def test_regex_basic_pattern() -> None:
    """Test basic regex pattern matching."""
    check = RegexMatching(
        text="The price is $10.99",
        pattern=r"\$\d+\.\d{2}",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.message is not None
    assert "matches the regex pattern" in result.message.lower()
    assert result.details["pattern"] == r"\$\d+\.\d{2}"


async def test_regex_pattern_not_found() -> None:
    """Test regex pattern that doesn't match."""
    check = RegexMatching(
        text="Hello World",
        pattern=r"\d+",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "does not match" in result.message.lower()


async def test_regex_with_start_anchor() -> None:
    """Test regex pattern with start anchor."""
    check = RegexMatching(
        text="Hello World",
        pattern=r"^Hello",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_regex_with_end_anchor() -> None:
    """Test regex pattern with end anchor."""
    check = RegexMatching(
        text="Hello World",
        pattern=r"World$",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_regex_with_word_boundaries() -> None:
    """Test regex pattern with word boundaries."""
    check_pass = RegexMatching(
        text="The city is Paris, France",
        pattern=r"\bParis\b",
    )
    result = await check_pass.run(Trace())
    assert result.status == CheckStatus.PASS

    # Should not match partial word
    check_fail = RegexMatching(
        text="The Parisian café",
        pattern=r"\bParis\b",
    )
    result = await check_fail.run(Trace())
    assert result.status == CheckStatus.FAIL


# Character classes and quantifiers
async def test_regex_character_classes() -> None:
    """Test regex with character classes for email validation."""
    check = RegexMatching(
        text="Contact: user@example.com",
        pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_regex_quantifiers() -> None:
    """Test regex with quantifiers."""
    check = RegexMatching(
        text="There are 42 items",
        pattern=r"\d{1,3}",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


# Case sensitivity with regex
async def test_regex_case_sensitive() -> None:
    """Test case-sensitive regex matching."""
    check = RegexMatching(
        text="Hello WORLD",
        pattern=r"world",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL


async def test_regex_case_insensitive() -> None:
    """Test case-insensitive regex matching using inline modifier."""
    check = RegexMatching(
        text="Hello WORLD",
        pattern=r"(?i)world",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


# Invalid regex patterns
async def test_regex_invalid_pattern() -> None:
    """Test behavior with invalid regex pattern."""
    check = RegexMatching(
        text="Hello World",
        pattern=r"[invalid(",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "invalid regex pattern" in result.message.lower()


async def test_regex_unclosed_group() -> None:
    """Test invalid regex with unclosed group."""
    check = RegexMatching(
        text="Test string",
        pattern=r"(unclosed",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "invalid regex pattern" in result.message.lower()


# Regex with whitespace
async def test_regex_with_multiple_whitespace() -> None:
    """Test that regex matches raw text with multiple spaces."""
    check = RegexMatching(
        text="Hello    World   Test",
        pattern=r"Hello\s+World\s+Test",
    )
    result = await check.run(Trace())
    # Regex matches raw text with multiple spaces
    assert result.status == CheckStatus.PASS


async def test_regex_with_unicode() -> None:
    """Test regex matching with Unicode characters."""
    check = RegexMatching(
        text="café is great",
        pattern=r"café.*great",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_regex_exact_whitespace() -> None:
    """Test regex with exact whitespace matching."""
    check = RegexMatching(
        text="Hello    World",
        pattern=r"Hello\s+World",
    )
    result = await check.run(Trace())
    # Matches raw text with multiple spaces
    assert result.status == CheckStatus.PASS


# Regex with trace extraction
async def test_regex_with_trace_extraction() -> None:
    """Test regex matching with pattern from trace."""
    check = RegexMatching(
        text_key="trace.last.outputs.response",
        pattern_key="trace.last.inputs.expected_pattern",
    )
    interaction = Interaction(
        inputs={"expected_pattern": r"\d{3}-\d{3}-\d{4}"},
        outputs={"response": "Call me at 555-123-4567"},
    )
    result = await check.run(Trace(interactions=[interaction]))
    assert result.status == CheckStatus.PASS
    assert result.details["pattern"] == r"\d{3}-\d{3}-\d{4}"


async def test_regex_extract_both_from_trace() -> None:
    """Test extracting both text and pattern from trace."""
    check = RegexMatching(
        text_key="trace.last.outputs.answer",
        pattern_key="trace.last.inputs.pattern",
    )
    interaction = Interaction(
        inputs={"pattern": r"^[A-Z]"},
        outputs={"answer": "Hello there"},
    )
    result = await check.run(Trace(interactions=[interaction]))
    assert result.status == CheckStatus.PASS


# Multiline and dotall flags
async def test_regex_multiline_flag() -> None:
    """Test regex with multiline inline modifier."""
    check = RegexMatching(
        text="Line 1\nLine 2\nLine 3",
        pattern=r"(?m)^Line 2$",
    )
    result = await check.run(Trace())
    # With (?m), ^ and $ match line boundaries
    assert result.status == CheckStatus.PASS


async def test_regex_dotall_flag() -> None:
    """Test regex with dotall inline modifier."""
    check = RegexMatching(
        text="Line 1\nLine 2",
        pattern=r"(?s)Line 1.Line 2",
    )
    result = await check.run(Trace())
    # With (?s), . matches newlines
    assert result.status == CheckStatus.PASS


async def test_regex_ascii_flag() -> None:
    """Test regex with ASCII inline modifier."""
    check = RegexMatching(
        text="café",
        pattern=r"(?a)\w+",
    )
    result = await check.run(Trace())
    # With (?a), \w only matches [a-zA-Z0-9_]
    # So it won't match the é
    assert result.status == CheckStatus.PASS  # Still matches "caf"


# Special regex features
async def test_regex_groups() -> None:
    """Test regex with capture groups."""
    check = RegexMatching(
        text="Date: 2024-01-15",
        pattern=r"(\d{4})-(\d{2})-(\d{2})",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_regex_alternation() -> None:
    """Test regex with alternation."""
    check = RegexMatching(
        text="I prefer Python",
        pattern=r"Python|Java|JavaScript",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_regex_inline_flags() -> None:
    """Test regex with inline flags."""
    check = RegexMatching(
        text="Hello WORLD",
        pattern=r"(?i)world",
    )
    result = await check.run(Trace())
    # Inline flag should work
    assert result.status == CheckStatus.PASS


# Edge cases
async def test_literal_special_chars_not_escaped() -> None:
    """Test that special characters are treated as regex in regex mode."""
    check = RegexMatching(
        text="The price is $10.99",
        pattern="$10.99",  # Without escaping, $ means end of string
    )
    result = await check.run(Trace())
    # This should fail because $ is treated as anchor
    assert result.status == CheckStatus.FAIL


async def test_empty_pattern_regex_mode() -> None:
    """Test behavior with empty pattern in regex mode."""
    check = RegexMatching(
        text="Hello",
        pattern="",
    )
    result = await check.run(Trace())
    # Empty regex matches any string
    assert result.status == CheckStatus.PASS


async def test_missing_pattern_validation() -> None:
    """Test that either pattern or pattern_key must be provided."""
    with pytest.raises(ValueError, match="pattern"):
        RegexMatching(text="Hello")


async def test_cannot_provide_both_pattern_and_pattern_key() -> None:
    """Test that providing both pattern and pattern_key raises an error."""
    with pytest.raises(ValueError, match="Exactly one"):
        RegexMatching(
            text="Hello World",
            pattern=r"hello",
            pattern_key="trace.last.inputs.key",
        )


# Real-world patterns
async def test_email_pattern() -> None:
    """Test email validation with regex."""
    check = RegexMatching(
        text="Contact me at john.doe@example.com for more info",
        pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_phone_number_pattern() -> None:
    """Test phone number matching."""
    check = RegexMatching(
        text="Call me at 555-123-4567",
        pattern=r"\d{3}-\d{3}-\d{4}",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


async def test_url_pattern() -> None:
    """Test URL matching."""
    check = RegexMatching(
        text="Visit https://example.com for details",
        pattern=r"https?://[^\s]+",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS


# Trace error handling
async def test_missing_pattern_in_trace() -> None:
    """Test error handling when pattern cannot be extracted from trace."""
    check = RegexMatching(
        text="Hello",
        pattern_key="trace.last.inputs.nonexistent",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "no value found for pattern" in result.message.lower()


async def test_missing_text_in_trace() -> None:
    """Test error handling when text cannot be extracted from trace."""
    check = RegexMatching(
        text_key="trace.last.outputs.nonexistent",
        pattern="test",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "no value found for text" in result.message.lower()
