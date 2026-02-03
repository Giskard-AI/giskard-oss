"""Tests for the normalization utility functions."""

import unicodedata

from giskard.checks.utils.normalization import normalize_data, normalize_string


def test_normalize_string_e_acute_nfc() -> None:
    """Test that normalize_string handles 'é' (U+00E9) with NFC normalization."""
    # U+00E9 is already in NFC form
    text = "café"  # Uses U+00E9
    result = normalize_string(text, "NFC")
    # Should remain the same (already normalized)
    assert result == "café"
    assert len(result) == 4  # c, a, f, é


def test_normalize_string_e_acute_nfd() -> None:
    """Test that normalize_string handles 'é' (U+0065 U+0301) with NFD normalization."""
    # U+0065 U+0301 is already in NFD form
    text = "caf\u0065\u0301"  # Uses U+0065 U+0301
    result = normalize_string(text, "NFD")
    # Should remain decomposed
    assert result == "caf\u0065\u0301"
    assert len(result) == 5  # c, a, f, e, combining acute


def test_normalize_string_e_acute_nfc_to_nfc() -> None:
    """Test that NFC normalization converts NFD 'é' to NFC."""
    # Input is in NFD form (U+0065 U+0301)
    text_nfd = "caf\u0065\u0301"  # Uses U+0065 U+0301
    result = normalize_string(text_nfd, "NFC")
    # Should convert to NFC form (U+00E9)
    assert result == "café"
    assert len(result) == 4  # c, a, f, é (composed)
    # Verify it's actually U+00E9
    assert unicodedata.normalize("NFC", result) == result


def test_normalize_string_e_acute_nfd_to_nfd() -> None:
    """Test that NFD normalization converts NFC 'é' to NFD."""
    # Input is in NFC form (U+00E9)
    text_nfc = "café"  # Uses U+00E9
    result = normalize_string(text_nfc, "NFD")
    # Should convert to NFD form (U+0065 U+0301)
    expected = "caf\u0065\u0301"
    assert result == expected
    assert len(result) == 5  # c, a, f, e, combining acute
    # Verify it's actually decomposed
    assert unicodedata.normalize("NFD", result) == result


def test_normalize_string_e_acute_nfc_equivalence() -> None:
    """Test that NFC normalization makes both forms equivalent."""
    # Both forms should normalize to the same NFC representation
    text_nfc = "café"  # Uses U+00E9
    text_nfd = "caf\u0065\u0301"  # Uses U+0065 U+0301

    result_nfc = normalize_string(text_nfc, "NFC")
    result_nfd = normalize_string(text_nfd, "NFC")

    # Both should be equal after NFC normalization
    assert result_nfc == result_nfd
    assert result_nfc == "café"


def test_normalize_string_e_acute_nfd_equivalence() -> None:
    """Test that NFD normalization makes both forms equivalent."""
    # Both forms should normalize to the same NFD representation
    text_nfc = "café"  # Uses U+00E9
    text_nfd = "caf\u0065\u0301"  # Uses U+0065 U+0301

    result_nfc = normalize_string(text_nfc, "NFD")
    result_nfd = normalize_string(text_nfd, "NFD")

    # Both should be equal after NFD normalization
    assert result_nfc == result_nfd
    assert result_nfd == "caf\u0065\u0301"


def test_normalize_string_e_acute_nfkc_equivalence() -> None:
    """Test that NFKC normalization makes both forms equivalent."""
    # Both forms should normalize to the same NFKC representation
    text_nfc = "café"  # Uses U+00E9
    text_nfd = "caf\u0065\u0301"  # Uses U+0065 U+0301

    result_nfc = normalize_string(text_nfc, "NFKC")
    result_nfd = normalize_string(text_nfd, "NFKC")

    # Both should be equal after NFKC normalization
    assert result_nfc == result_nfd


def test_normalize_string_e_acute_no_normalization() -> None:
    """Test that without normalization, different forms remain different."""
    # Without normalization, different Unicode representations should remain different
    text_nfc = "café"  # Uses U+00E9
    text_nfd = "caf\u0065\u0301"  # Uses U+0065 U+0301

    result_nfc = normalize_string(text_nfc, None)
    result_nfd = normalize_string(text_nfd, None)

    # Without normalization, they should NOT be equal
    assert result_nfc != result_nfd
    assert result_nfc == "café"
    assert result_nfd == "caf\u0065\u0301"


def test_normalize_string_whitespace_preserved() -> None:
    """Test that whitespace normalization still works with 'é'."""
    text = "  café  "  # Uses U+00E9
    result = normalize_string(text, "NFC")
    # Whitespace should be normalized (trimmed and collapsed)
    assert result == "café"


def test_normalize_data_string_e_acute() -> None:
    """Test that normalize_data handles 'é' in string values."""
    text_nfc = "café"  # Uses U+00E9
    text_nfd = "caf\u0065\u0301"  # Uses U+0065 U+0301

    result_nfc = normalize_data(text_nfc, "NFC")
    result_nfd = normalize_data(text_nfd, "NFC")

    # Both should normalize to the same form
    assert result_nfc == result_nfd
    assert result_nfc == "café"


def test_normalize_data_dict_e_acute() -> None:
    """Test that normalize_data handles 'é' in dictionary values."""
    data = {
        "text1": "café",  # Uses U+00E9
        "text2": "caf\u0065\u0301",  # Uses U+0065 U+0301
    }

    result = normalize_data(data, "NFC")

    # Both values should normalize to NFC form
    assert result["text1"] == "café"
    assert result["text2"] == "café"
    assert result["text1"] == result["text2"]


def test_normalize_data_list_e_acute() -> None:
    """Test that normalize_data handles 'é' in list values."""
    data = [
        "café",  # Uses U+00E9
        "caf\u0065\u0301",  # Uses U+0065 U+0301
    ]

    result = normalize_data(data, "NFC")

    # Both values should normalize to NFC form
    assert result[0] == "café"
    assert result[1] == "café"
    assert result[0] == result[1]
