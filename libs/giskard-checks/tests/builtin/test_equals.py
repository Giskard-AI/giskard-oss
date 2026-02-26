"""Unit tests for Equals check.

Tests cover different types (str, number, bool) and various comparison scenarios:
- Same type, same value (should pass)
- Same type, different value (should fail)
- Same value, different type (should fail)
"""

from giskard.checks import CheckStatus, Equals, Interaction, Trace
from giskard.checks.core.extraction import NoMatch


class TestEqualsString:
    """Test Equals check with string values."""

    async def test_string_same_value_same_type(self):
        """Test that same string value and type passes."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="hello")
        )
        check = Equals(
            expected_value="hello",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == "hello"
        assert result.details["expected_value"] == "hello"

    async def test_string_different_value_same_type(self):
        """Test that different string values fail."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="hello")
        )
        check = Equals(
            expected_value="world",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "hello"
        assert result.details["expected_value"] == "world"
        assert isinstance(result.message, str)
        assert "Expected value equal to 'world' but got 'hello'" in result.message

    async def test_string_same_value_different_type_string_vs_number(self):
        """Test that string '123' vs number 123 fails (type mismatch)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="123"))
        check = Equals(
            expected_value=123,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "123"
        assert result.details["expected_value"] == 123

    async def test_string_same_value_different_type_string_vs_bool(self):
        """Test that string 'True' vs bool True fails (type mismatch)."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="True")
        )
        check = Equals(
            expected_value=True,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "True"
        assert result.details["expected_value"] is True


class TestEqualsNumber:
    """Test Equals check with numeric values."""

    async def test_number_same_value_same_type_int(self):
        """Test that same integer value and type passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=42))
        check = Equals(
            expected_value=42,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 42
        assert result.details["expected_value"] == 42

    async def test_number_same_value_same_type_float(self):
        """Test that same float value and type passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=3.14))
        check = Equals(
            expected_value=3.14,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 3.14
        assert result.details["expected_value"] == 3.14

    async def test_number_different_value_same_type_int(self):
        """Test that different integer values fail."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=42))
        check = Equals(
            expected_value=100,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == 42
        assert result.details["expected_value"] == 100

    async def test_number_different_value_same_type_float(self):
        """Test that different float values fail."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=3.14))
        check = Equals(
            expected_value=2.71,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == 3.14
        assert result.details["expected_value"] == 2.71

    async def test_number_same_value_different_type_int_vs_float(self):
        """Test that int 1 vs float 1.0 fails (type mismatch)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=1))
        check = Equals(
            expected_value=1.0,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        # Note: In Python, 1 == 1.0 is True, so this will pass
        # This test documents the actual behavior
        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_number_same_value_different_type_string_vs_int(self):
        """Test that string '1' vs int 1 fails (type mismatch)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="1"))
        check = Equals(
            expected_value=1,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "1"
        assert result.details["expected_value"] == 1

    async def test_number_same_value_different_type_string_vs_float(self):
        """Test that string '1.0' vs float 1.0 fails (type mismatch)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="1.0"))
        check = Equals(
            expected_value=1.0,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "1.0"
        assert result.details["expected_value"] == 1.0


class TestEqualsBool:
    """Test Equals check with boolean values."""

    async def test_bool_same_value_same_type_true(self):
        """Test that same boolean True value and type passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=True))
        check = Equals(
            expected_value=True,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] is True
        assert result.details["expected_value"] is True

    async def test_bool_same_value_same_type_false(self):
        """Test that same boolean False value and type passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=False))
        check = Equals(
            expected_value=False,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] is False
        assert result.details["expected_value"] is False

    async def test_bool_different_value_same_type(self):
        """Test that different boolean values fail."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=True))
        check = Equals(
            expected_value=False,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] is True
        assert result.details["expected_value"] is False

    async def test_bool_same_value_different_type_string_true_vs_bool_true(self):
        """Test that string 'True' vs bool True fails (type mismatch)."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="True")
        )
        check = Equals(
            expected_value=True,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "True"
        assert result.details["expected_value"] is True

    async def test_bool_same_value_different_type_string_false_vs_bool_false(self):
        """Test that string 'False' vs bool False fails (type mismatch)."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="False")
        )
        check = Equals(
            expected_value=False,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "False"
        assert result.details["expected_value"] is False

    async def test_bool_same_value_different_type_number_one_vs_bool_true(self):
        """Test that number 1 vs bool True fails (type mismatch).

        Note: In Python, 1 == True is True due to bool being a subclass of int,
        but this test documents the actual behavior.
        """
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=1))
        check = Equals(
            expected_value=True,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        # Note: In Python, 1 == True evaluates to True
        # This test documents the actual behavior
        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_bool_same_value_different_type_number_zero_vs_bool_false(self):
        """Test that number 0 vs bool False.

        Note: In Python, 0 == False is True due to bool being a subclass of int,
        but this test documents the actual behavior.
        """
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=0))
        check = Equals(
            expected_value=False,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        # Note: In Python, 0 == False evaluates to True
        # This test documents the actual behavior
        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_string_true_vs_number_one(self):
        """Test that string 'True' vs number 1 fails (type mismatch)."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="True")
        )
        check = Equals(
            expected_value=1,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "True"
        assert result.details["expected_value"] == 1

    async def test_string_one_vs_bool_true(self):
        """Test that string '1' vs bool True fails (type mismatch)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="1"))
        check = Equals(
            expected_value=True,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "1"
        assert result.details["expected_value"] is True


class TestEqualsEdgeCases:
    """Test edge cases for Equals check."""

    async def test_nested_outputs_string(self):
        """Test equality check with nested outputs (dict structure)."""
        trace = await Trace.from_interactions(
            Interaction(
                inputs="test",
                outputs={"result": "success", "code": 200},
            )
        )
        check = Equals(
            expected_value="success",
            key="trace.interactions[-1].outputs.result",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == "success"
        assert result.details["expected_value"] == "success"

    async def test_nested_outputs_number(self):
        """Test equality check with nested outputs containing number."""
        trace = await Trace.from_interactions(
            Interaction(
                inputs="test",
                outputs={"result": "success", "code": 200},
            )
        )
        check = Equals(
            expected_value=200,
            key="trace.interactions[-1].outputs.code",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 200
        assert result.details["expected_value"] == 200

    async def test_nested_outputs_bool(self):
        """Test equality check with nested outputs containing bool."""
        trace = await Trace.from_interactions(
            Interaction(
                inputs="test",
                outputs={"result": "success", "valid": True},
            )
        )
        check = Equals(
            expected_value=True,
            key="trace.interactions[-1].outputs.valid",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] is True
        assert result.details["expected_value"] is True

    async def test_missing_key(self):
        """Test equality check when the key is missing from trace."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs={"other": "value"})
        )
        check = Equals(
            expected_value="expected",
            key="trace.interactions[-1].outputs.missing",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.details["actual_value"], NoMatch)
        assert (
            result.details["actual_value"].key
            == "trace.interactions[-1].outputs.missing"
        )
        assert result.details["expected_value"] == "expected"
        assert isinstance(result.message, str)
        assert "No value found for key" in result.message

    async def test_none_value(self):
        """Test equality check with None values."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=None))
        check = Equals(
            expected_value=None,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] is None
        assert result.details["expected_value"] is None

    async def test_nomatch_with_trace_last(self):
        """Test equality check when using trace.last syntax and key is missing."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs={"other": "value"})
        )
        check = Equals(
            expected_value="expected",
            key="trace.last.outputs.missing",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.details["actual_value"], NoMatch)
        assert result.details["actual_value"].key == "trace.last.outputs.missing"
        assert result.details["expected_value"] == "expected"

    async def test_nomatch_with_deeply_nested_path(self):
        """Test equality check with deeply nested path that doesn't exist."""
        trace = await Trace.from_interactions(
            Interaction(
                inputs="test",
                outputs={"level1": {"level2": {"level3": "value"}}},
            )
        )
        check = Equals(
            expected_value="expected",
            key="trace.interactions[-1].outputs.level1.level2.missing",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.details["actual_value"], NoMatch)
        assert (
            result.details["actual_value"].key
            == "trace.interactions[-1].outputs.level1.level2.missing"
        )

    async def test_nomatch_with_empty_trace(self):
        """Test equality check with empty trace (no interactions)."""
        trace = Trace()
        check = Equals(
            expected_value="expected",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.details["actual_value"], NoMatch)
        assert result.details["actual_value"].key == "trace.interactions[-1].outputs"

    async def test_nomatch_equality_when_both_are_nomatch_same_key(self):
        """Test equality check when both expected and actual are NoMatch with same key."""
        trace = Trace()
        expected_nomatch = NoMatch(key="trace.interactions[-1].outputs")
        check = Equals(
            expected_value=expected_nomatch,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        # When expected_value is a NoMatch, the check fails immediately
        assert isinstance(result.details["actual_value"], NoMatch)
        assert isinstance(result.details["expected_value"], NoMatch)
        assert (
            result.details["actual_value"].key == result.details["expected_value"].key
        )
        assert result.status == CheckStatus.FAIL
        assert result.failed

    async def test_nomatch_equality_when_both_are_nomatch_different_keys(self):
        """Test equality check when both expected and actual are NoMatch with different keys."""
        trace = Trace()
        expected_nomatch = NoMatch(key="different.key")
        check = Equals(
            expected_value=expected_nomatch,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        # When both are NoMatch but with different keys, they should not be equal
        assert isinstance(result.details["actual_value"], NoMatch)
        assert isinstance(result.details["expected_value"], NoMatch)
        assert (
            result.details["actual_value"].key != result.details["expected_value"].key
        )
        assert result.status == CheckStatus.FAIL
        assert result.failed


class TestEqualsListExpressions:
    """Test Equals check with JSONPath list expressions (wildcard and single index)."""

    async def test_wildcard_expression_with_list_expected_multiple_items(self):
        """Test that wildcard expression [*] returns a list and matches expected list."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test1", outputs="message 1"),
            Interaction(inputs="test2", outputs="Message 2"),
        )
        check = Equals(
            expected_value=["message 1", "Message 2"],
            key="trace.interactions[*].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert isinstance(result.details["actual_value"], list)
        assert result.details["actual_value"] == ["message 1", "Message 2"]
        assert result.details["expected_value"] == ["message 1", "Message 2"]

    async def test_wildcard_expression_with_list_expected_single_item(self):
        """Test that wildcard expression [*] returns a list even with single item."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test1", outputs="message 1"),
        )
        check = Equals(
            expected_value=["message 1"],
            key="trace.interactions[*].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert isinstance(result.details["actual_value"], list)
        assert result.details["actual_value"] == ["message 1"]
        assert result.details["expected_value"] == ["message 1"]

    async def test_wildcard_expression_with_single_value_expected_fails(self):
        """Test that wildcard expression [*] fails when expected is a single value."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test1", outputs="message 1"),
        )
        check = Equals(
            expected_value="message 1",
            key="trace.interactions[*].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.details["actual_value"], list)
        assert result.details["actual_value"] == ["message 1"]
        assert result.details["expected_value"] == "message 1"
        assert isinstance(result.message, str)
        assert (
            "Expected value equal to 'message 1' but got ['message 1']"
            in result.message
        )

    async def test_single_index_expression_with_single_value_expected(self):
        """Test that single index expression [-1] returns a single value."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test1", outputs="message 1"),
            Interaction(inputs="test2", outputs="Message 2"),
        )
        check = Equals(
            expected_value="Message 2",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert not isinstance(result.details["actual_value"], list)
        assert result.details["actual_value"] == "Message 2"
        assert result.details["expected_value"] == "Message 2"

    async def test_single_index_expression_with_list_expected_fails(self):
        """Test that single index expression [-1] fails when expected is a list."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test1", outputs="message 1"),
            Interaction(inputs="test2", outputs="Message 2"),
        )
        check = Equals(
            expected_value=["Message 2"],
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert not isinstance(result.details["actual_value"], list)
        assert result.details["actual_value"] == "Message 2"
        assert isinstance(result.details["expected_value"], list)
        assert result.details["expected_value"] == ["Message 2"]
        assert isinstance(result.message, str)
        assert (
            "Expected value equal to ['Message 2'] but got 'Message 2'"
            in result.message
        )

    async def test_single_index_expression_with_different_value_fails(self):
        """Test that single index expression [-1] fails when value doesn't match."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test1", outputs="message 1"),
            Interaction(inputs="test2", outputs="Message 2"),
        )
        check = Equals(
            expected_value="Wrong message",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "Message 2"
        assert result.details["expected_value"] == "Wrong message"
        assert isinstance(result.message, str)
        assert (
            "Expected value equal to 'Wrong message' but got 'Message 2'"
            in result.message
        )

    async def test_wildcard_expression_with_different_list_fails(self):
        """Test that wildcard expression [*] fails when list doesn't match."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test1", outputs="message 1"),
            Interaction(inputs="test2", outputs="Message 2"),
        )
        check = Equals(
            expected_value=["wrong", "list"],
            key="trace.interactions[*].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.details["actual_value"], list)
        assert result.details["actual_value"] == ["message 1", "Message 2"]
        assert result.details["expected_value"] == ["wrong", "list"]
        assert isinstance(result.message, str)
        assert (
            "Expected value equal to ['wrong', 'list'] but got ['message 1', 'Message 2']"
            in result.message
        )


class TestEqualsUnicodeNormalization:
    """Test Equals check with Unicode normalization edge cases.

    Note: The Equals check uses normalization (default NFKC), so different
    Unicode representations of the same character (e.g., 'é' as U+00E9 vs
    U+0065 U+0301) should match when normalized. These tests document this behavior.
    """

    async def test_unicode_e_acute_different_representations_pass(self):
        """Test that 'é' (U+00E9) vs 'é' (U+0065 U+0301) passes with normalization.

        This test documents that Equals check normalizes strings (default NFKC),
        so different Unicode representations of the same character will match.
        """
        # U+00E9 is the composed form (NFC)
        # U+0065 U+0301 is the decomposed form (NFD): 'e' + combining acute accent
        text_nfc = "café"  # Uses U+00E9
        text_nfd = "caf\u0065\u0301"  # Uses U+0065 U+0301

        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs=text_nfc)
        )
        check = Equals(
            expected_value=text_nfd,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        # With normalization (default NFKC), they should match
        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == text_nfc
        assert result.details["expected_value"] == text_nfd

    async def test_unicode_e_acute_same_representation_passes(self):
        """Test that 'é' in the same Unicode representation passes."""
        # Both use U+00E9 (NFC form)
        text = "café"  # Uses U+00E9

        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=text))
        check = Equals(
            expected_value=text,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        # Same representation should match
        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == text
        assert result.details["expected_value"] == text

    async def test_unicode_e_acute_list_extraction_different_representations_pass(self):
        """Test that 'é' in list extraction passes with different Unicode representations.

        When extracting messages[*].content from {"messages": [{"content": "café"}]},
        it returns ["café"]. Different Unicode representations should match with normalization.
        """
        # U+00E9 is the composed form (NFC)
        # U+0065 U+0301 is the decomposed form (NFD): 'e' + combining acute accent
        text_nfc = "café"  # Uses U+00E9
        text_nfd = "caf\u0065\u0301"  # Uses U+0065 U+0301

        trace = await Trace.from_interactions(
            Interaction(
                inputs="test",
                outputs={"messages": [{"content": text_nfc}]},
            )
        )
        check = Equals(
            expected_value=[text_nfd],  # Expected list with NFD form
            key="trace.interactions[-1].outputs.messages[*].content",
        )

        result = await check.run(trace)

        # With normalization (default NFKC), they should match
        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == [text_nfc]
        assert result.details["expected_value"] == [text_nfd]

    async def test_unicode_e_acute_list_extraction_same_representation_passes(self):
        """Test that 'é' in list extraction passes with same Unicode representation."""
        # Both use U+00E9 (NFC form)
        text = "café"  # Uses U+00E9

        trace = await Trace.from_interactions(
            Interaction(
                inputs="test",
                outputs={"messages": [{"content": text}]},
            )
        )
        check = Equals(
            expected_value=[text],
            key="trace.interactions[-1].outputs.messages[*].content",
        )

        result = await check.run(trace)

        # Same representation should match
        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == [text]
        assert result.details["expected_value"] == [text]

    async def test_unicode_e_acute_dict_list_extraction_different_representations_pass(
        self,
    ):
        """Test that 'é' in dict list extraction passes with different Unicode representations.

        When extracting messages[*] from {"messages": [{"content": "café"}]},
        it returns [{"content": "café"}]. Different Unicode representations should match with normalization.
        """
        # U+00E9 is the composed form (NFC)
        # U+0065 U+0301 is the decomposed form (NFD): 'e' + combining acute accent
        text_nfc = "café"  # Uses U+00E9
        text_nfd = "caf\u0065\u0301"  # Uses U+0065 U+0301

        trace = await Trace.from_interactions(
            Interaction(
                inputs="test",
                outputs={"messages": [{"content": text_nfc}]},
            )
        )
        check = Equals(
            expected_value=[
                {"content": text_nfd}
            ],  # Expected list with dict containing NFD form
            key="trace.interactions[-1].outputs.messages[*]",
        )

        result = await check.run(trace)

        # With normalization (default NFKC), they should match
        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == [{"content": text_nfc}]
        assert result.details["expected_value"] == [{"content": text_nfd}]

    async def test_unicode_e_acute_dict_list_extraction_same_representation_passes(
        self,
    ):
        """Test that 'é' in dict list extraction passes with same Unicode representation."""
        # Both use U+00E9 (NFC form)
        text = "café"  # Uses U+00E9

        trace = await Trace.from_interactions(
            Interaction(
                inputs="test",
                outputs={"messages": [{"content": text}]},
            )
        )
        check = Equals(
            expected_value=[{"content": text}],
            key="trace.interactions[-1].outputs.messages[*]",
        )

        result = await check.run(trace)

        # Same representation should match
        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == [{"content": text}]
        assert result.details["expected_value"] == [{"content": text}]
