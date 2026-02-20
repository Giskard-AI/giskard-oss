from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self, override

from giskard.core import NOT_PROVIDED, NotProvided, provide_not_none
from pydantic import Field, model_validator

from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, provided_or_resolve, resolve
from ..core.result import CheckResult
from ..core.trace import Trace
from ..utils.normalization import NormalizationForm, normalize_data


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
    expected_value: ExpectedType | None = Field(
        default=None,
        description="The expected value to compare against. If None, the expected value is extracted from the trace using the expected_value_key.",
    )
    expected_value_key: JSONPathStr | NotProvided = Field(
        default=NOT_PROVIDED,
        description="The key to extract the expected value from the trace. If None, the expected value is used as is. If provided, the expected value is extracted from the trace using the expected_value_key.",
    )
    normalization_form: NormalizationForm | None = Field(
        default="NFKC",
        description="Unicode normalization form to apply before comparison. Defaults to NFKC.",
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
        if isinstance(self.expected_value, NotProvided) and isinstance(
            self.expected_value_key, NotProvided
        ):
            raise ValueError(
                "Either 'expected_value' or 'expected_value_key' must be provided"
            )
        return self

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the check against the provided trace."""
        actual_value = resolve(trace, self.key)
        expected_value = provided_or_resolve(
            trace,
            key=provide_not_none(self.expected_value_key),
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

        normalized_actual_value = normalize_data(actual_value, self.normalization_form)
        normalized_expected_value = normalize_data(
            expected_value, self.normalization_form
        )
        try:
            if self._compare(normalized_actual_value, normalized_expected_value):
                return CheckResult.success(
                    message=f"The actual value {repr(actual_value)} is {self._comparison_message} the expected value {repr(expected_value)}.",
                    details=details,
                )
        except Exception:
            return CheckResult.failure(
                message=f"Comparison not supported: {type(actual_value).__name__} does not support {self._operator_symbol} comparison with {type(expected_value).__name__}",
                details=details,
            )

        return CheckResult.failure(
            message=f"Expected value {self._comparison_message} {repr(expected_value)} but got {repr(actual_value)}",
            details=details,
        )


@Check.register("lesser_than")
class LesserThan[InputType, OutputType, TraceType: Trace, ExpectedType](  # pyright: ignore[reportMissingTypeArgument]
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


@Check.register("lesser_than_equals")
class LesserThanEquals[InputType, OutputType, TraceType: Trace, ExpectedType](  # pyright: ignore[reportMissingTypeArgument]
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
