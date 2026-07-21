import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal, Self, override

from pydantic import Field, model_validator
from pydantic.experimental.missing_sentinel import MISSING

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, provided_or_resolve, resolve
from ..core.result import CheckResult
from ..utils.normalization import NormalizationForm, normalize_data

MatchMode = Literal["any", "all", "none"]
type MatchCollection[T] = list[T] | set[T] | tuple[T, ...]


class ComparisonCheck[InputType, OutputType, TraceType: Trace, ExpectedType](  # pyright: ignore[reportMissingTypeArgument]
    ABC, Check[InputType, OutputType, TraceType]
):
    """Base class for comparison checks.

    This abstract base class implements the common logic for comparison checks
    using the Template Method pattern. Subclasses must implement:

    - `_compare()`: Performs the actual comparison operation (e.g., ``<``, ``>``)
    - `_comparison_message`: Returns a human-readable description of the comparison
      (e.g., "less than", "greater than or equal to")
    - `_operator_symbol`: Returns the operator symbol for technical error messages
      (e.g., ``"<"``, ``">"``, ``"<="``)

    The base class handles:
    - Value extraction from traces
    - NoMatch handling
    - Error handling for unsupported comparisons
    - Result formatting
    """

    key: JSONPathStr = Field(
        ..., description="The key to extract the actual value from the trace"
    )
    expected_value: ExpectedType | MISSING = Field(
        default=MISSING,
        description="The expected value to compare against. If omitted, the expected value is extracted from the trace using expected_value_key. Explicit None is valid and compares against None.",
    )
    expected_value_key: JSONPathStr | MISSING = Field(
        default=MISSING,
        description="The key to extract the expected value from the trace. If omitted, use expected_value directly. If provided, the expected value is extracted from the trace using this key.",
    )
    normalization_form: NormalizationForm | None = Field(
        default="NFKC",
        description="Unicode normalization form to apply before comparison. Defaults to NFKC.",
    )
    match: MatchMode | MISSING = Field(
        default=MISSING,
        description=(
            "How to apply the comparison when the resolved actual value is a collection. "
            "When omitted, the resolved value is compared directly. "
            "'any' passes if at least one item matches, 'all' if every item matches, "
            "'none' if no item matches. Requires a list, set, or tuple."
        ),
    )

    @abstractmethod
    def _compare(self, actual_value: Any, expected_value: ExpectedType) -> bool:
        """Compare the actual value with the expected value."""
        ...

    @property
    @abstractmethod
    def _comparison_message(self) -> str:
        """Get the human-readable comparison message (e.g., 'less than')."""
        ...

    @property
    @abstractmethod
    def _operator_symbol(self) -> str:
        """Get the operator symbol (e.g., '<', '>', '<=') for error messages."""
        ...

    @model_validator(mode="after")
    def validate_expected_value_or_expected_value_key(self) -> Self:
        """Validate that exactly one of expected_value or expected_value_key is provided."""
        if (self.expected_value is MISSING) == (self.expected_value_key is MISSING):
            raise ValueError(
                "Exactly one of 'expected_value' or 'expected_value_key' must be provided"
            )
        return self

    def _compare_normalized(
        self, normalized_actual: Any, normalized_expected: ExpectedType
    ) -> bool | None:
        try:
            return self._compare(normalized_actual, normalized_expected)
        except Exception:
            return None

    def _try_compare(
        self, actual_value: Any, expected_value: ExpectedType
    ) -> bool | None:
        return self._compare_normalized(
            normalize_data(actual_value, self.normalization_form),
            normalize_data(expected_value, self.normalization_form),
        )

    def _try_compare_to_normalized_expected(
        self, actual_value: Any, normalized_expected: ExpectedType
    ) -> bool | None:
        return self._compare_normalized(
            normalize_data(actual_value, self.normalization_form),
            normalized_expected,
        )

    def _unsupported_comparison_message(
        self, actual_value: Any, expected_value: ExpectedType
    ) -> str:
        return (
            f"Comparison not supported: items in {type(actual_value).__name__} "
            f"do not support {self._operator_symbol} comparison with "
            f"{type(expected_value).__name__}"
        )

    def _collection_match_message(
        self,
        passed: bool,
        actual_value: Any,
        expected_value: ExpectedType,
        matched_items: list[Any] | None = None,
    ) -> str:
        actual_repr = repr(actual_value)
        expected_repr = repr(expected_value)
        comparison = self._comparison_message
        if self.match == "any":
            if passed:
                return f"At least one value in {actual_repr} is {comparison} {expected_repr}."
            return (
                f"Expected at least one value {comparison} {expected_repr} "
                f"but none matched in {actual_repr}."
            )
        if self.match == "all":
            if passed:
                return f"All values in {actual_repr} are {comparison} {expected_repr}."
            return f"Expected all values {comparison} {expected_repr} but got {actual_repr}."
        if passed:
            return f"No value in {actual_repr} is {comparison} {expected_repr}."
        return (
            f"Expected no value {comparison} {expected_repr} "
            f"but found matches in {repr(matched_items)}."
        )

    def _run_collection_match(
        self,
        actual_value: Any,
        expected_value: ExpectedType,
        details: dict[str, Any],
    ) -> CheckResult:
        if not isinstance(actual_value, (list, set, tuple)):
            return CheckResult.failure(
                message=(
                    f"Expected a list, set, or tuple at key '{self.key}' when match is "
                    f"{self.match!r}, but got {type(actual_value).__name__}."
                ),
                details=details,
            )

        collection: MatchCollection[Any] = actual_value
        normalized_expected = normalize_data(expected_value, self.normalization_form)
        comparison_results: list[bool | None] = []
        matched_items: list[Any] = []
        for item in collection:
            result = self._try_compare_to_normalized_expected(item, normalized_expected)
            comparison_results.append(result)
            if result:
                matched_items.append(item)

        if any(result is None for result in comparison_results):
            return CheckResult.failure(
                message=self._unsupported_comparison_message(
                    actual_value, expected_value
                ),
                details=details,
            )

        if self.match == "any":
            passed = any(comparison_results)
        elif self.match == "all":
            passed = all(comparison_results)
        else:
            passed = not any(comparison_results)

        if passed:
            return CheckResult.success(
                message=self._collection_match_message(
                    passed, actual_value, expected_value
                ),
                details=details,
            )

        return CheckResult.failure(
            message=self._collection_match_message(
                passed,
                actual_value,
                expected_value,
                matched_items if self.match == "none" else None,
            ),
            details=details,
        )

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the check against the provided trace."""
        actual_value = resolve(trace, self.key)
        expected_value = provided_or_resolve(
            trace,
            key=self.expected_value_key,
            value=self.expected_value,
        )

        details = {
            "actual_value": actual_value,
            "expected_value": expected_value,
        }

        if isinstance(expected_value, NoMatch):
            return CheckResult.failure(
                message=f"No value found for expected value key '{self.expected_value_key}'.",
                details=details,
            )

        if isinstance(actual_value, NoMatch):
            return CheckResult.failure(
                message=f"No value found for key '{self.key}', expected a value {self._comparison_message} {repr(self.expected_value)}.",
                details=details,
            )

        if self.match is not MISSING:
            return self._run_collection_match(actual_value, expected_value, details)

        compare_result = self._try_compare(actual_value, expected_value)
        if compare_result is None:
            return CheckResult.failure(
                message=f"Comparison not supported: {type(actual_value).__name__} does not support {self._operator_symbol} comparison with {type(expected_value).__name__}",
                details=details,
            )
        if compare_result:
            return CheckResult.success(
                message=f"The actual value {repr(actual_value)} is {self._comparison_message} the expected value {repr(expected_value)}.",
                details=details,
            )

        return CheckResult.failure(
            message=f"Expected value {self._comparison_message} {repr(expected_value)} but got {repr(actual_value)}",
            details=details,
        )


@Check.register("less_than")
class LessThan[InputType, OutputType, TraceType: Trace, ExpectedType](  # pyright: ignore[reportMissingTypeArgument]
    ComparisonCheck[InputType, OutputType, TraceType, ExpectedType]
):
    """Check that validates if extracted values are less than an expected value.

    This check extracts values from a trace and compares them against a
    specified expected value using Python's ``__lt__`` method.

    .. warning::
        For object instances, this check uses Python's ``__lt__`` method for
        comparison. The behavior depends on how the object's ``__lt__`` method
        is implemented. For custom objects, ensure that ``__lt__`` is properly
        defined to match your comparison requirements. If the comparison is not
        supported (e.g., incompatible types or missing method), the check will
        return a failure result.

    Attributes
    ----------
    expected_value : ExpectedType
        The expected value to compare against the extracted values
    key : str
        The key to extract the actual value from the trace
    """

    @override
    def _compare(self, actual_value: Any, expected_value: ExpectedType) -> bool:
        """Compare the actual value with the expected value."""
        return actual_value < expected_value

    @property
    @override
    def _comparison_message(self) -> str:
        """Get the human-readable comparison message."""
        return "less than"

    @property
    @override
    def _operator_symbol(self) -> str:
        """Get the operator symbol for error messages."""
        return "<"


@Check.register("lesser_than")
class LesserThan[InputType, OutputType, TraceType: Trace, ExpectedType](  # pyright: ignore[reportMissingTypeArgument]
    LessThan[InputType, OutputType, TraceType, ExpectedType]
):
    """Deprecated alias for :class:`LessThan`.

    .. deprecated::
        Use :class:`LessThan` instead. This alias remains for backward
        compatibility with serialized checks using ``kind="lesser_than"``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "LesserThan is deprecated; use LessThan instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


@Check.register("greater_than")
class GreaterThan[InputType, OutputType, TraceType: Trace, ExpectedType](  # pyright: ignore[reportMissingTypeArgument]
    ComparisonCheck[InputType, OutputType, TraceType, ExpectedType]
):
    """Check that validates if extracted values are greater than an expected value.

    This check extracts values from a trace and compares them against a
    specified expected value using Python's ``__gt__`` method.

    .. warning::
        For object instances, this check uses Python's ``__gt__`` method for
        comparison. The behavior depends on how the object's ``__gt__`` method
        is implemented. For custom objects, ensure that ``__gt__`` is properly
        defined to match your comparison requirements. If the comparison is not
        supported (e.g., incompatible types or missing method), the check will
        return a failure result.

    Attributes
    ----------
    expected_value : ExpectedType
        The expected value to compare against the extracted values
    key : str
        The key to extract the actual value from the trace
    """

    @override
    def _compare(self, actual_value: Any, expected_value: ExpectedType) -> bool:
        """Compare the actual value with the expected value."""
        return actual_value > expected_value

    @property
    @override
    def _comparison_message(self) -> str:
        """Get the human-readable comparison message."""
        return "greater than"

    @property
    @override
    def _operator_symbol(self) -> str:
        """Get the operator symbol for error messages."""
        return ">"


@Check.register("less_than_equals")
class LessThanEquals[InputType, OutputType, TraceType: Trace, ExpectedType](  # pyright: ignore[reportMissingTypeArgument]
    ComparisonCheck[InputType, OutputType, TraceType, ExpectedType]
):
    """Check that validates if extracted values are less than or equal to an expected value.

    This check extracts values from a trace and compares them against a
    specified expected value using Python's ``__le__`` method.

    .. warning::
        For object instances, this check uses Python's ``__le__`` method for
        comparison. The behavior depends on how the object's ``__le__`` method
        is implemented. For custom objects, ensure that ``__le__`` is properly
        defined to match your comparison requirements. If the comparison is not
        supported (e.g., incompatible types or missing method), the check will
        return a failure result.

    Attributes
    ----------
    expected_value : ExpectedType
        The expected value to compare against the extracted values
    key : str
        The key to extract the actual value from the trace
    """

    @override
    def _compare(self, actual_value: Any, expected_value: ExpectedType) -> bool:
        """Compare the actual value with the expected value."""
        return actual_value <= expected_value

    @property
    @override
    def _comparison_message(self) -> str:
        """Get the human-readable comparison message."""
        return "less than or equal to"

    @property
    @override
    def _operator_symbol(self) -> str:
        """Get the operator symbol for error messages."""
        return "<="


@Check.register("lesser_than_equals")
class LesserThanEquals[InputType, OutputType, TraceType: Trace, ExpectedType](  # pyright: ignore[reportMissingTypeArgument]
    LessThanEquals[InputType, OutputType, TraceType, ExpectedType]
):
    """Deprecated alias for :class:`LessThanEquals`.

    .. deprecated::
        Use :class:`LessThanEquals` instead. This alias remains for backward
        compatibility with serialized checks using ``kind="lesser_than_equals"``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "LesserThanEquals is deprecated; use LessThanEquals instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


@Check.register("greater_than_equals")
class GreaterEquals[InputType, OutputType, TraceType: Trace, ExpectedType](  # pyright: ignore[reportMissingTypeArgument]
    ComparisonCheck[InputType, OutputType, TraceType, ExpectedType]
):
    """Check that validates if extracted values are greater than or equal to an expected value.

    This check extracts values from a trace and compares them against a
    specified expected value using Python's ``__ge__`` method.

    .. warning::
        For object instances, this check uses Python's ``__ge__`` method for
        comparison. The behavior depends on how the object's ``__ge__`` method
        is implemented. For custom objects, ensure that ``__ge__`` is properly
        defined to match your comparison requirements. If the comparison is not
        supported (e.g., incompatible types or missing method), the check will
        return a failure result.

    Attributes
    ----------
    expected_value : ExpectedType
        The expected value to compare against the extracted values
    key : str
        The key to extract the actual value from the trace
    """

    @override
    def _compare(self, actual_value: Any, expected_value: ExpectedType) -> bool:
        """Compare the actual value with the expected value."""
        return actual_value >= expected_value

    @property
    @override
    def _comparison_message(self) -> str:
        """Get the human-readable comparison message."""
        return "greater than or equal to"

    @property
    @override
    def _operator_symbol(self) -> str:
        """Get the operator symbol for error messages."""
        return ">="


@Check.register("equals")
class Equals[InputType, OutputType, TraceType: Trace, ExpectedType](  # pyright: ignore[reportMissingTypeArgument]
    ComparisonCheck[InputType, OutputType, TraceType, ExpectedType]
):
    """Check that validates if extracted values equal an expected value.

    This check extracts values from a trace and compares them against a
    specified expected value using Python's ``__eq__`` method.

    .. warning::
        For object instances, this check uses Python's ``__eq__`` method for
        comparison. The behavior depends on how the object's ``__eq__`` method
        is implemented. For custom objects, ensure that ``__eq__`` is properly
        defined to match your comparison requirements. If the comparison is not
        supported (e.g., incompatible types or missing method), the check will
        return a failure result.

    Attributes
    ----------
    expected_value : ExpectedType
        The expected value to compare against the extracted values
    key : str
        The key to extract the actual value from the trace
    """

    @override
    def _compare(self, actual_value: Any, expected_value: ExpectedType) -> bool:
        """Compare the actual value with the expected value."""
        return actual_value == expected_value

    @property
    @override
    def _comparison_message(self) -> str:
        """Get the human-readable comparison message."""
        return "equal to"

    @property
    @override
    def _operator_symbol(self) -> str:
        """Get the operator symbol for error messages."""
        return "=="


@Check.register("not_equals")
class NotEquals[InputType, OutputType, TraceType: Trace, ExpectedType](  # pyright: ignore[reportMissingTypeArgument]
    ComparisonCheck[InputType, OutputType, TraceType, ExpectedType]
):
    """Check that validates if extracted values do not equal an expected value.

    This check extracts values from a trace and compares them against a
    specified expected value using Python's ``__ne__`` method.

    .. warning::
        For object instances, this check uses Python's ``__ne__`` method for
        comparison. The behavior depends on how the object's ``__ne__`` method
        is implemented. For custom objects, ensure that ``__ne__`` is properly
        defined to match your comparison requirements. If the comparison is not
        supported (e.g., incompatible types or missing method), the check will
        return a failure result.

    Attributes
    ----------
    expected_value : ExpectedType
        The expected value to compare against the extracted values
    key : str
        The key to extract the actual value from the trace
    """

    @override
    def _compare(self, actual_value: Any, expected_value: ExpectedType) -> bool:
        """Compare the actual value with the expected value."""
        return actual_value != expected_value

    @property
    @override
    def _comparison_message(self) -> str:
        """Get the human-readable comparison message."""
        return "not equal to"

    @property
    @override
    def _operator_symbol(self) -> str:
        """Get the operator symbol for error messages."""
        return "!="
