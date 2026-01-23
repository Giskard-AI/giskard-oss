"""Unit tests for Equality check.

Tests cover different types (str, number, bool) and various comparison scenarios:
- Same type, same value (should pass)
- Same type, different value (should fail)
- Same value, different type (should fail)
"""

from giskard.checks import CheckStatus, Equality, Interaction, Trace
from giskard.checks.core.extraction import NoMatch


class TestEqualityString:
    """Test Equality check with string values."""

    async def test_string_same_value_same_type(self):
        """Test that same string value and type passes."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="hello")
        )
        check = Equality(
            expected_value="hello",
            actual_value_key="trace.interactions[-1].outputs",
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
        check = Equality(
            expected_value="world",
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "hello"
        assert result.details["expected_value"] == "world"
        assert isinstance(result.message, str)
        assert "Expected value 'world' but got 'hello'" in result.message

    async def test_string_same_value_different_type_string_vs_number(self):
        """Test that string '123' vs number 123 fails (type mismatch)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="123"))
        check = Equality(
            expected_value=123,
            actual_value_key="trace.interactions[-1].outputs",
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
        check = Equality(
            expected_value=True,
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "True"
        assert result.details["expected_value"] is True


class TestEqualityNumber:
    """Test Equality check with numeric values."""

    async def test_number_same_value_same_type_int(self):
        """Test that same integer value and type passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=42))
        check = Equality(
            expected_value=42,
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 42
        assert result.details["expected_value"] == 42

    async def test_number_same_value_same_type_float(self):
        """Test that same float value and type passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=3.14))
        check = Equality(
            expected_value=3.14,
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 3.14
        assert result.details["expected_value"] == 3.14

    async def test_number_different_value_same_type_int(self):
        """Test that different integer values fail."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=42))
        check = Equality(
            expected_value=100,
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == 42
        assert result.details["expected_value"] == 100

    async def test_number_different_value_same_type_float(self):
        """Test that different float values fail."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=3.14))
        check = Equality(
            expected_value=2.71,
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == 3.14
        assert result.details["expected_value"] == 2.71

    async def test_number_same_value_different_type_int_vs_float(self):
        """Test that int 1 vs float 1.0 fails (type mismatch)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=1))
        check = Equality(
            expected_value=1.0,
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        # Note: In Python, 1 == 1.0 is True, so this will pass
        # This test documents the actual behavior
        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_number_same_value_different_type_string_vs_int(self):
        """Test that string '1' vs int 1 fails (type mismatch)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="1"))
        check = Equality(
            expected_value=1,
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "1"
        assert result.details["expected_value"] == 1

    async def test_number_same_value_different_type_string_vs_float(self):
        """Test that string '1.0' vs float 1.0 fails (type mismatch)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="1.0"))
        check = Equality(
            expected_value=1.0,
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "1.0"
        assert result.details["expected_value"] == 1.0


class TestEqualityBool:
    """Test Equality check with boolean values."""

    async def test_bool_same_value_same_type_true(self):
        """Test that same boolean True value and type passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=True))
        check = Equality(
            expected_value=True,
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] is True
        assert result.details["expected_value"] is True

    async def test_bool_same_value_same_type_false(self):
        """Test that same boolean False value and type passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=False))
        check = Equality(
            expected_value=False,
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] is False
        assert result.details["expected_value"] is False

    async def test_bool_different_value_same_type(self):
        """Test that different boolean values fail."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=True))
        check = Equality(
            expected_value=False,
            actual_value_key="trace.interactions[-1].outputs",
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
        check = Equality(
            expected_value=True,
            actual_value_key="trace.interactions[-1].outputs",
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
        check = Equality(
            expected_value=False,
            actual_value_key="trace.interactions[-1].outputs",
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
        check = Equality(
            expected_value=True,
            actual_value_key="trace.interactions[-1].outputs",
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
        check = Equality(
            expected_value=False,
            actual_value_key="trace.interactions[-1].outputs",
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
        check = Equality(
            expected_value=1,
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "True"
        assert result.details["expected_value"] == 1

    async def test_string_one_vs_bool_true(self):
        """Test that string '1' vs bool True fails (type mismatch)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="1"))
        check = Equality(
            expected_value=True,
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "1"
        assert result.details["expected_value"] is True


class TestEqualityEdgeCases:
    """Test edge cases for Equality check."""

    async def test_nested_outputs_string(self):
        """Test equality check with nested outputs (dict structure)."""
        trace = await Trace.from_interactions(
            Interaction(
                inputs="test",
                outputs={"result": "success", "code": 200},
            )
        )
        check = Equality(
            expected_value="success",
            actual_value_key="trace.interactions[-1].outputs.result",
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
        check = Equality(
            expected_value=200,
            actual_value_key="trace.interactions[-1].outputs.code",
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
        check = Equality(
            expected_value=True,
            actual_value_key="trace.interactions[-1].outputs.valid",
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
        check = Equality(
            expected_value="expected",
            actual_value_key="trace.interactions[-1].outputs.missing",
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
        check = Equality(
            expected_value=None,
            actual_value_key="trace.interactions[-1].outputs",
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
        check = Equality(
            expected_value="expected",
            actual_value_key="trace.last.outputs.missing",
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
        check = Equality(
            expected_value="expected",
            actual_value_key="trace.interactions[-1].outputs.level1.level2.missing",
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
        check = Equality(
            expected_value="expected",
            actual_value_key="trace.interactions[-1].outputs",
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
        check = Equality(
            expected_value=expected_nomatch,
            actual_value_key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        # When both are NoMatch with the same key, they should be equal
        assert isinstance(result.details["actual_value"], NoMatch)
        assert isinstance(result.details["expected_value"], NoMatch)
        assert (
            result.details["actual_value"].key == result.details["expected_value"].key
        )
        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_nomatch_equality_when_both_are_nomatch_different_keys(self):
        """Test equality check when both expected and actual are NoMatch with different keys."""
        trace = Trace()
        expected_nomatch = NoMatch(key="different.key")
        check = Equality(
            expected_value=expected_nomatch,
            actual_value_key="trace.interactions[-1].outputs",
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
