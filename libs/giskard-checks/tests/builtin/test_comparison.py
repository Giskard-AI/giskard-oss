"""Unit tests for comparison checks (LessThan, GreaterThan, LessThanEquals, GreaterEquals).

Tests cover different types (numbers, strings) and various comparison scenarios:
- Success cases (e.g., 5 < 10 should pass for LessThan)
- Failure cases (e.g., 10 < 5 should fail for LessThan)
- TypeError handling (missing methods and incompatible types)
"""

import warnings
from typing import Any

import pytest
from giskard.checks import (
    Check,
    CheckStatus,
    Equals,
    GreaterEquals,
    GreaterThan,
    Interaction,
    LesserThan,
    LesserThanEquals,
    LessThan,
    LessThanEquals,
    NotEquals,
    Trace,
)
from giskard.checks.core.extraction import NoMatch


class TestLessThan:
    """Test LessThan check."""

    async def test_number_lesser_than_success(self):
        """Test that 5 < 10 passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=5))
        check = LessThan(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 5
        assert result.details["expected_value"] == 10

    async def test_number_lesser_than_failure(self):
        """Test that 10 < 5 fails."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=10))
        check = LessThan(
            expected_value=5,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == 10
        assert result.details["expected_value"] == 5
        assert isinstance(result.message, str)
        assert "Expected value less than 5 but got 10" in result.message

    async def test_number_lesser_than_equal_fails(self):
        """Test that 5 < 5 fails (equal values)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=5))
        check = LessThan(
            expected_value=5,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == 5
        assert result.details["expected_value"] == 5

    async def test_float_lesser_than_success(self):
        """Test that 3.14 < 5.0 passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=3.14))
        check = LessThan(
            expected_value=5.0,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_string_lesser_than_success(self):
        """Test that 'apple' < 'banana' passes."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="apple")
        )
        check = LessThan(
            expected_value="banana",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_string_lesser_than_failure(self):
        """Test that 'banana' < 'apple' fails."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="banana")
        )
        check = LessThan(
            expected_value="apple",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed

    async def test_missing_key(self):
        """Test LessThan check when the key is missing from trace."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs={"other": "value"})
        )
        check = LessThan(
            expected_value=10,
            key="trace.interactions[-1].outputs.missing",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.details["actual_value"], NoMatch)
        assert result.message is not None

    async def test_nested_outputs(self):
        """Test LessThan check with nested outputs."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs={"value": 5})
        )
        check = LessThan(
            expected_value=10,
            key="trace.interactions[-1].outputs.value",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 5

    async def test_typeerror_incompatible_types(self):
        """Test LessThan with incompatible types (string vs int)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="5"))
        check = LessThan(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "5"
        assert result.details["expected_value"] == 10
        assert result.message is not None

    async def test_typeerror_missing_method(self):
        """Test LessThan with object that doesn't implement __lt__."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs=object())
        )
        check = LessThan(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.message is not None
        assert "Comparison not supported" in result.message
        assert "< comparison" in result.message


class TestGreaterThan:
    """Test GreaterThan check."""

    async def test_number_greater_than_success(self):
        """Test that 10 > 5 passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=10))
        check = GreaterThan(
            expected_value=5,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 10
        assert result.details["expected_value"] == 5

    async def test_number_greater_than_failure(self):
        """Test that 5 > 10 fails."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=5))
        check = GreaterThan(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == 5
        assert result.details["expected_value"] == 10
        assert isinstance(result.message, str)
        assert "Expected value greater than 10 but got 5" in result.message

    async def test_number_greater_than_equal_fails(self):
        """Test that 5 > 5 fails (equal values)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=5))
        check = GreaterThan(
            expected_value=5,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed

    async def test_string_greater_than_success(self):
        """Test that 'banana' > 'apple' passes."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="banana")
        )
        check = GreaterThan(
            expected_value="apple",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_typeerror_incompatible_types(self):
        """Test GreaterThan with incompatible types (string vs int)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="10"))
        check = GreaterThan(
            expected_value=5,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.message is not None
        assert "Comparison not supported" in result.message
        assert "str" in result.message
        assert "int" in result.message
        assert "> comparison" in result.message

    async def test_typeerror_missing_method(self):
        """Test GreaterThan with object that doesn't implement __gt__."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs=object())
        )
        check = GreaterThan(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.message is not None
        assert "Comparison not supported" in result.message
        assert "> comparison" in result.message


class TestLessThanEquals:
    """Test LessThanEquals check."""

    async def test_number_lesser_than_equals_success_less(self):
        """Test that 5 <= 10 passes (less than case)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=5))
        check = LessThanEquals(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 5
        assert result.details["expected_value"] == 10

    async def test_number_lesser_than_equals_success_equal(self):
        """Test that 5 <= 5 passes (equal case)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=5))
        check = LessThanEquals(
            expected_value=5,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 5
        assert result.details["expected_value"] == 5

    async def test_number_lesser_than_equals_failure(self):
        """Test that 10 <= 5 fails."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=10))
        check = LessThanEquals(
            expected_value=5,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == 10
        assert result.details["expected_value"] == 5
        assert isinstance(result.message, str)
        assert "Expected value less than or equal to 5 but got 10" in result.message

    async def test_string_lesser_than_equals_success(self):
        """Test that 'apple' <= 'banana' passes."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="apple")
        )
        check = LessThanEquals(
            expected_value="banana",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_string_lesser_than_equals_equal(self):
        """Test that 'apple' <= 'apple' passes."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="apple")
        )
        check = LessThanEquals(
            expected_value="apple",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_typeerror_incompatible_types(self):
        """Test LessThanEquals with incompatible types (string vs int)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="5"))
        check = LessThanEquals(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.message is not None
        assert "Comparison not supported" in result.message
        assert "str" in result.message
        assert "int" in result.message
        assert "<= comparison" in result.message

    async def test_typeerror_missing_method(self):
        """Test LessThanEquals with object that doesn't implement __le__."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs=object())
        )
        check = LessThanEquals(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.message is not None
        assert "Comparison not supported" in result.message
        assert "<= comparison" in result.message


class TestGreaterEquals:
    """Test GreaterEquals check."""

    async def test_number_greater_equals_success_greater(self):
        """Test that 10 >= 5 passes (greater than case)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=10))
        check = GreaterEquals(
            expected_value=5,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 10
        assert result.details["expected_value"] == 5

    async def test_number_greater_equals_success_equal(self):
        """Test that 5 >= 5 passes (equal case)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=5))
        check = GreaterEquals(
            expected_value=5,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 5
        assert result.details["expected_value"] == 5

    async def test_number_greater_equals_failure(self):
        """Test that 5 >= 10 fails."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=5))
        check = GreaterEquals(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == 5
        assert result.details["expected_value"] == 10
        assert isinstance(result.message, str)
        assert "Expected value greater than or equal to 10 but got 5" in result.message

    async def test_string_greater_equals_success(self):
        """Test that 'banana' >= 'apple' passes."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="banana")
        )
        check = GreaterEquals(
            expected_value="apple",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_string_greater_equals_equal(self):
        """Test that 'apple' >= 'apple' passes."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="apple")
        )
        check = GreaterEquals(
            expected_value="apple",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_typeerror_incompatible_types(self):
        """Test GreaterEquals with incompatible types (string vs int)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="10"))
        check = GreaterEquals(
            expected_value=5,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.message is not None
        assert "Comparison not supported" in result.message
        assert "str" in result.message
        assert "int" in result.message
        assert ">= comparison" in result.message

    async def test_typeerror_missing_method(self):
        """Test GreaterEquals with object that doesn't implement __ge__."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs=object())
        )
        check = GreaterEquals(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.message is not None
        assert "Comparison not supported" in result.message
        assert ">= comparison" in result.message


class TestComparisonEdgeCases:
    """Test edge cases for comparison checks."""

    async def test_none_value_lesser_than(self):
        """Test LessThan with None values."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=None))
        check = LessThan(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        # None comparisons raise TypeError in Python
        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.message is not None
        assert "Comparison not supported" in result.message

    async def test_none_value_greater_than(self):
        """Test GreaterThan with None values."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=None))
        check = GreaterThan(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.message is not None
        assert "Comparison not supported" in result.message

    async def test_list_vs_string_incompatible(self):
        """Test comparison with list vs string (incompatible types)."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs=[1, 2, 3])
        )
        check = LessThan(
            expected_value="abc",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.message is not None
        assert "Comparison not supported" in result.message
        assert "list" in result.message
        assert "str" in result.message

    async def test_custom_class_with_comparison(self):
        """Test comparison with custom class that implements comparison for its own type."""

        class ComparableValue:
            def __init__(self, value: int):
                self.value = value

            def __lt__(self, other: "ComparableValue") -> bool:
                return self.value < other.value

            def __gt__(self, other: "ComparableValue") -> bool:
                return self.value > other.value

            def __le__(self, other: "ComparableValue") -> bool:
                return self.value <= other.value

            def __ge__(self, other: "ComparableValue") -> bool:
                return self.value >= other.value

        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs=ComparableValue(5))
        )
        check = LessThan(
            expected_value=ComparableValue(10),
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_custom_class_incompatible_with_int(self):
        """Test comparison with custom class that doesn't support comparison with int."""

        class ComparableValue:
            def __init__(self, value: int):
                self.value = value

            def __lt__(self, other: "ComparableValue") -> bool:
                return self.value < other.value

        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs=ComparableValue(5))
        )
        check = LessThan(
            expected_value=10,  # int, not ComparableValue
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.message is not None
        assert "Comparison not supported" in result.message

    async def test_wildcard_expression_with_list(self):
        """Test LessThan with wildcard expression returning a list."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test1", outputs=5),
            Interaction(inputs="test2", outputs=3),
        )
        check = LessThan(
            expected_value=[10, 10],  # Expected list
            key="trace.interactions[*].outputs",
        )

        result = await check.run(trace)

        # Lists can be compared, but [5, 3] < [10, 10] should pass
        assert result.status == CheckStatus.PASS
        assert result.passed
        assert isinstance(result.details["actual_value"], list)

    async def test_single_index_expression(self):
        """Test LessThan with single index expression."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test1", outputs=5),
            Interaction(inputs="test2", outputs=15),
        )
        check = LessThan(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == 15


class TestNotEquals:
    """Test NotEquals check."""

    async def test_number_not_equals_success(self):
        """Test that 5 != 10 passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=5))
        check = NotEquals(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 5
        assert result.details["expected_value"] == 10

    async def test_number_not_equals_failure(self):
        """Test that 5 != 5 fails (equal values)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=5))
        check = NotEquals(
            expected_value=5,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == 5
        assert result.details["expected_value"] == 5
        assert isinstance(result.message, str)
        assert "Expected value not equal to 5 but got 5" in result.message

    async def test_float_not_equals_success(self):
        """Test that 3.14 != 5.0 passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=3.14))
        check = NotEquals(
            expected_value=5.0,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_string_not_equals_success(self):
        """Test that 'hello' != 'world' passes."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="hello")
        )
        check = NotEquals(
            expected_value="world",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == "hello"
        assert result.details["expected_value"] == "world"

    async def test_string_not_equals_failure(self):
        """Test that 'hello' != 'hello' fails (equal values)."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="hello")
        )
        check = NotEquals(
            expected_value="hello",
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == "hello"
        assert result.details["expected_value"] == "hello"

    async def test_bool_not_equals_success(self):
        """Test that True != False passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=True))
        check = NotEquals(
            expected_value=False,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] is True
        assert result.details["expected_value"] is False

    async def test_bool_not_equals_failure(self):
        """Test that True != True fails (equal values)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=True))
        check = NotEquals(
            expected_value=True,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] is True
        assert result.details["expected_value"] is True

    async def test_different_types_string_vs_int_success(self):
        """Test that '5' != 5 passes (different types)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="5"))
        check = NotEquals(
            expected_value=5,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == "5"
        assert result.details["expected_value"] == 5

    async def test_different_types_string_vs_bool_success(self):
        """Test that 'True' != True passes (different types)."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="True")
        )
        check = NotEquals(
            expected_value=True,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == "True"
        assert result.details["expected_value"] is True

    async def test_missing_key(self):
        """Test NotEquals check when the key is missing from trace."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs={"other": "value"})
        )
        check = NotEquals(
            expected_value=10,
            key="trace.interactions[-1].outputs.missing",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.details["actual_value"], NoMatch)
        assert result.message is not None

    async def test_nested_outputs(self):
        """Test NotEquals check with nested outputs."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs={"value": 5})
        )
        check = NotEquals(
            expected_value=10,
            key="trace.interactions[-1].outputs.value",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == 5

    async def test_none_value_not_equals_success(self):
        """Test that None != 10 passes."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=None))
        check = NotEquals(
            expected_value=10,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] is None
        assert result.details["expected_value"] == 10

    async def test_none_value_not_equals_failure(self):
        """Test that None != None fails (equal values)."""
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs=None))
        check = NotEquals(
            expected_value=None,
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] is None
        assert result.details["expected_value"] is None

    async def test_list_not_equals_success(self):
        """Test that [1, 2] != [3, 4] passes."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs=[1, 2])
        )
        check = NotEquals(
            expected_value=[3, 4],
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == [1, 2]
        assert result.details["expected_value"] == [3, 4]

    async def test_list_not_equals_failure(self):
        """Test that [1, 2] != [1, 2] fails (equal values)."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs=[1, 2])
        )
        check = NotEquals(
            expected_value=[1, 2],
            key="trace.interactions[-1].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["actual_value"] == [1, 2]
        assert result.details["expected_value"] == [1, 2]


class TestComparisonSentinelDefault:
    """Regression tests for issue #2501: omitting expected_value must raise an error."""

    @pytest.mark.parametrize(
        "check_cls",
        [Equals, GreaterThan, LessThan, GreaterEquals, LessThanEquals, NotEquals],
    )
    def test_omitting_both_raises(self, check_cls):
        """Omitting both expected_value and expected_value_key must raise ValueError."""
        with pytest.raises(ValueError, match="expected_value"):
            check_cls(key="trace.last.outputs")

    def test_explicit_none_is_valid(self):
        """explicit expected_value=None must be accepted (compares against None)."""
        check = Equals(key="trace.last.outputs", expected_value=None)
        assert check.expected_value is None

    def test_expected_value_key_is_valid(self):
        """Providing expected_value_key without expected_value must be accepted."""
        check = Equals(
            key="trace.last.outputs",
            expected_value_key="trace.last.metadata.expected",
        )
        assert check.expected_value_key == "trace.last.metadata.expected"

    def test_cannot_provide_both_expected_value_and_expected_value_key(self):
        """Providing both expected_value and expected_value_key must raise ValueError."""
        with pytest.raises(ValueError, match="Exactly one"):
            Equals(
                key="trace.last.outputs",
                expected_value=42,
                expected_value_key="trace.last.metadata.expected",
            )


class TestComparisonMatchMode:
    """Test collection match modes (any, all, none) on ComparisonCheck."""

    @staticmethod
    async def _tool_calls_trace(
        tool_calls: list[dict[str, object]] | None = None,
    ) -> Trace[Any, Any]:
        if tool_calls is None:
            tool_calls = [
                {"name": "search", "args": {}},
                {"name": "summarize", "args": {}},
            ]
        return await Trace.from_interactions(
            Interaction(
                inputs="test",
                outputs="result",
                metadata={"tool_calls": tool_calls},
            )
        )

    async def test_equals_match_any_with_wildcard_path(self):
        """match='any' checks whether any list item equals the expected scalar."""
        trace = await self._tool_calls_trace()
        check = Equals(
            expected_value="search",
            key="trace.last.metadata.tool_calls[*].name",
            match="any",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["actual_value"] == ["search", "summarize"]

    async def test_equals_match_any_fails_when_none_match(self):
        trace = await self._tool_calls_trace()
        check = Equals(
            expected_value="delete",
            key="trace.last.metadata.tool_calls[*].name",
            match="any",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.message, str)
        assert "none matched" in result.message

    async def test_equals_match_all_passes_when_all_match(self):
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="ok"),
            Interaction(inputs="test", outputs="ok"),
        )
        check = Equals(
            expected_value="ok",
            key="trace.interactions[*].outputs",
            match="all",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_equals_match_all_fails_when_one_differs(self):
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="ok"),
            Interaction(inputs="test", outputs="nope"),
        )
        check = Equals(
            expected_value="ok",
            key="trace.interactions[*].outputs",
            match="all",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.message, str)
        assert "Expected all values equal to 'ok'" in result.message

    async def test_equals_match_none_passes_when_no_item_matches(self):
        trace = await self._tool_calls_trace()
        check = Equals(
            expected_value="delete",
            key="trace.last.metadata.tool_calls[*].name",
            match="none",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_equals_match_none_fails_when_item_matches(self):
        trace = await self._tool_calls_trace()
        check = Equals(
            expected_value="search",
            key="trace.last.metadata.tool_calls[*].name",
            match="none",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.message, str)
        assert "found matches" in result.message

    async def test_match_any_fails_on_scalar_value(self):
        trace = await Trace.from_interactions(Interaction(inputs="test", outputs="ok"))
        check = Equals(
            expected_value="ok",
            key="trace.last.outputs",
            match="any",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.message, str)
        assert "Expected a list, set, or tuple" in result.message

    async def test_match_any_fails_on_empty_collection(self):
        trace = await self._tool_calls_trace(tool_calls=[])
        check = Equals(
            expected_value="search",
            key="trace.last.metadata.tool_calls[*].name",
            match="any",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.message, str)
        assert "none matched" in result.message

    @pytest.mark.parametrize(
        ("match", "expect_pass"),
        [("all", True), ("none", True)],
        ids=["all", "none"],
    )
    async def test_empty_collection_match_modes(self, match, expect_pass):
        trace = await self._tool_calls_trace(tool_calls=[])
        check = Equals(
            expected_value="search",
            key="trace.last.metadata.tool_calls[*].name",
            match=match,
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS if expect_pass else CheckStatus.FAIL
        assert result.passed if expect_pass else result.failed

    async def test_match_any_works_with_tuple(self):
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs=(1, 2, 3)),
        )
        check = GreaterThan(
            expected_value=2,
            key="trace.last.outputs",
            match="any",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_match_any_works_with_set(self):
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs={1, 2, 3}),
        )
        check = GreaterThan(
            expected_value=2,
            key="trace.last.outputs",
            match="any",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    @pytest.mark.parametrize("match", ["any", "all", "none"])
    async def test_partial_unsupported_comparison_fails(self, match):
        """Unsupported comparisons on any item fail instead of being ignored."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs=[5, "abc"]),
        )
        check = GreaterThan(
            expected_value=3,
            key="trace.last.outputs",
            match=match,
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.message, str)
        assert "Comparison not supported" in result.message

    async def test_match_any_passes_when_supported_items_match(self):
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs=[5, 1]),
        )
        check = GreaterThan(
            expected_value=3,
            key="trace.last.outputs",
            match="any",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_default_match_preserves_scalar_comparison(self):
        """Without match mode, wildcard paths still compare the full list."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="message 1"),
        )
        check = Equals(
            expected_value="message 1",
            key="trace.interactions[*].outputs",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert isinstance(result.message, str)
        assert (
            "Expected value equal to 'message 1' but got ['message 1']"
            in result.message
        )

    async def test_default_match_fixes_wildcard_with_match_any(self):
        trace = await Trace.from_interactions(
            Interaction(inputs="test", outputs="message 1"),
        )
        check = Equals(
            expected_value="message 1",
            key="trace.interactions[*].outputs",
            match="any",
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed


class TestLessThanBackwardCompat:
    """Backward compatibility for deprecated LesserThan names and kind strings."""

    def test_less_than_serialises_with_new_kind(self):
        check = LessThan(expected_value=10, key="trace.last.outputs")
        assert check.model_dump()["kind"] == "less_than"

    def test_lesser_than_serialises_with_legacy_kind(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            check = LesserThan(expected_value=10, key="trace.last.outputs")
        assert len(caught) == 1
        assert "LesserThan is deprecated" in str(caught[0].message)
        assert check.model_dump()["kind"] == "lesser_than"

    def test_lesser_than_kind_deserialises(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            original = LesserThan(expected_value=10, key="trace.last.outputs")
        restored = Check.model_validate(original.model_dump())
        assert isinstance(restored, LesserThan)
        assert restored.kind == "lesser_than"

    def test_less_than_equals_serialises_with_new_kind(self):
        check = LessThanEquals(expected_value=10, key="trace.last.outputs")
        assert check.model_dump()["kind"] == "less_than_equals"

    def test_lesser_than_equals_kind_deserialises(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            original = LesserThanEquals(expected_value=10, key="trace.last.outputs")
        restored = Check.model_validate(original.model_dump())
        assert isinstance(restored, LesserThanEquals)
        assert restored.kind == "lesser_than_equals"
