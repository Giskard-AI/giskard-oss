"""Text matching check implementations.

This module provides checks for text matching:
- StringMatching: Literal substring matching with normalization
- RegexMatching: Regular expression pattern matching
- ContainsAny: Validates if at least one value from a list appears within a text string
- ContainsAll: Validates if all values from a list appear within a text string
"""

from abc import ABC, abstractmethod
from typing import Any, Self, override

import regex
from pydantic import Field, model_validator
from pydantic.experimental.missing_sentinel import MISSING

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, provided_or_resolve
from ..core.result import CheckResult
from ..utils.normalization import NormalizationForm, normalize_string


class TextBasedCheck[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType], ABC
):
    """Base class for checks that validate text against a target value.

    This abstract class handles the common pattern of:
    1. Extracting text from a trace or direct value
    2. Extracting a target value (keyword/pattern) from trace or direct value
    3. Validating both values exist and are strings
    4. Delegating to subclass for specific matching logic

    Attributes
    ----------
    text : str | MISSING
        The text string to search within. If omitted, extracted from
        trace using ``text_key``.
    text_key : JSONPathStr
        JSONPath expression to extract the text from the trace. Defaults to
        "trace.last.outputs" which extracts the last interaction's outputs.
    """

    text: str | MISSING = Field(
        default=MISSING,
        description="The text string to search within.",
    )
    text_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract text from trace.",
    )

    def _extract_and_validate(
        self,
        trace: TraceType,
        target_value: str | MISSING,
        target_key: JSONPathStr | MISSING,
        target_name: str,
    ) -> tuple[str, str, dict[str, Any]] | tuple[None, None, CheckResult]:
        """Extract and validate text and target from trace or direct values.

        Parameters
        ----------
        trace : TraceType
            The trace to extract values from.
        target_value : str | MISSING
            Direct target value (keyword/pattern).
        target_key : JSONPathStr | MISSING
            JSONPath key to extract target from trace.
        target_name : str
            Name of the target parameter (for error messages).

        Returns
        -------
        tuple[str, str, dict] | tuple[None, None, CheckResult]
            Either (text, target, details) on success, or (None, None, error_result) on failure.
        """
        # Extract text and target
        text = provided_or_resolve(trace, key=self.text_key, value=self.text)
        target = provided_or_resolve(
            trace,
            key=target_key,
            value=target_value,
        )

        details = {"text": text, target_name: target}

        # Validate target
        if isinstance(target, NoMatch):
            return (
                None,
                None,
                CheckResult.failure(
                    message=f"No value found for {target_name} key '{target_key}'.",
                    details=details,
                ),
            )

        if not isinstance(target, str):
            return (
                None,
                None,
                CheckResult.failure(
                    message=f"Value for {target_name} is not a string, expected string but got {type(target).__name__}.",
                    details=details,
                ),
            )

        # Validate text
        if isinstance(text, NoMatch):
            return (
                None,
                None,
                CheckResult.failure(
                    message=f"No value found for text key '{self.text_key}'.",
                    details=details,
                ),
            )

        if not isinstance(text, str):
            return (
                None,
                None,
                CheckResult.failure(
                    message=f"Value for text is not a string, expected string but got {type(text).__name__}.",
                    details=details,
                ),
            )

        return text, target, details

    @abstractmethod
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the check. Must be implemented by subclasses."""
        ...


@Check.register("string_matching")
class StringMatching[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    TextBasedCheck[InputType, OutputType, TraceType]
):
    """Check that validates if a keyword appears within a text string.

    This check performs substring matching between a keyword and text, with
    support for Unicode normalization and case sensitivity control. Both the
    text and keyword can be provided directly or extracted from a trace using
    JSONPath expressions.

    The matching process:
    1. Extracts text and keyword (from provided values or trace)
    2. Applies Unicode normalization if specified
    3. Normalizes case if case-insensitive matching is enabled
    4. Normalizes whitespace (collapses multiple spaces, trims)
    5. Checks if the formatted keyword appears in the formatted text

    Attributes
    ----------
    text : str | MISSING
        The text string to search within. If omitted, extracted from
        trace using ``text_key``.
    text_key : JSONPathStr
        JSONPath expression to extract the text from the trace. Defaults to
        "trace.last.outputs" which extracts the last interaction's outputs.
    keyword : str | MISSING
        The keyword to search for within the text. If omitted, must provide
        ``keyword_key`` to extract from trace.
    keyword_key : JSONPathStr | MISSING
        JSONPath expression to extract the keyword from the trace. Either
        ``keyword`` or ``keyword_key`` must be provided.
    normalization_form : NormalizationForm | None
        Unicode normalization form to apply before matching. Options:
        - "NFC": Canonical Composition
        - "NFD": Canonical Decomposition
        - "NFKC": Compatibility Composition (default)
        - "NFKD": Compatibility Decomposition
        If None, no normalization is applied. Defaults to "NFKC".
    case_sensitive : bool
        If True, matching is case-sensitive. If False, both text and keyword
        are converted to lowercase before comparison. Defaults to True.

    Examples
    --------
    Direct text and keyword::

        check = StringMatching(
            text="Hello World",
            keyword="world",
            case_sensitive=False
        )

    Extract text from trace::

        check = StringMatching(
            keyword="Paris",
            text_key="trace.last.outputs.response"
        )

    Extract both from trace::

        check = StringMatching(
            text_key="trace.last.outputs.answer",
            keyword_key="trace.last.inputs.expected_keyword"
        )
    """

    keyword: str | MISSING = Field(
        default=MISSING,
        description="The keyword to search for within the text. Either this or keyword_key must be provided.",
    )
    keyword_key: JSONPathStr | MISSING = Field(
        default=MISSING,
        description="JSONPath expression to extract the keyword from the trace (e.g., 'trace.last.inputs.expected'). Either this or keyword must be provided.",
    )
    normalization_form: NormalizationForm | None = Field(
        default="NFKC",
        description="Unicode normalization form to apply (NFC, NFD, NFKC, NFKD). Defaults to NFKC.",
    )
    case_sensitive: bool = Field(
        default=True,
        description="If True, matching is case-sensitive. If False, both strings are lowercased before comparison.",
    )

    @model_validator(mode="after")
    def validate_keyword_or_keyword_key(self) -> Self:
        """Validate that exactly one of keyword or keyword_key is provided.

        Returns
        -------
        Self
            The validated instance.

        Raises
        ------
        ValueError
            If neither or both keyword and keyword_key are provided.
        """
        if (self.keyword is MISSING) == (self.keyword_key is MISSING):
            raise ValueError(
                "Exactly one of 'keyword' or 'keyword_key' must be provided, not both or neither."
            )

        return self

    def _format_str(self, value: str) -> str:
        """Format a string for matching by applying normalization and case handling.

        This method:
        1. Applies Unicode normalization if specified
        2. Converts to lowercase if case-insensitive matching is enabled
        3. Normalizes whitespace (collapses multiple spaces to single space, trims)

        Parameters
        ----------
        value : str
            The string to format.

        Returns
        -------
        str
            The formatted string ready for comparison.
        """
        value = normalize_string(value, self.normalization_form)

        if not self.case_sensitive:
            value = value.lower()

        return value

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the string matching check.

        Extracts text and keyword (from provided values or trace), formats them,
        and checks if the keyword appears within the text.

        Parameters
        ----------
        trace : TraceType
            The trace containing interaction history. Used to extract text/keyword
            if not provided directly.

        Returns
        -------
        CheckResult
            Success if keyword is found in text, failure otherwise. Includes
            details about the text, keyword, normalization form, and case sensitivity.
        """
        # Extract and validate using base class
        result = self._extract_and_validate(
            trace, self.keyword, self.keyword_key, "keyword"
        )

        # Check if extraction failed
        if result[0] is None:
            # Error occurred during extraction, return the error result
            return result[2]  # type: ignore[return-value]

        # Extract successful values
        text, keyword, details = result[0], result[1], result[2]

        # Add StringMatching-specific details
        details["normalization_form"] = self.normalization_form
        details["case_sensitive"] = self.case_sensitive

        # Format both strings for comparison
        formatted_text = self._format_str(text)
        formatted_keyword = self._format_str(keyword)

        # Check if keyword appears in text
        if formatted_keyword in formatted_text:
            return CheckResult.success(
                message=f"The answer contains the keyword '{keyword}'.",
                details=details,
            )

        return CheckResult.failure(
            message=f"The answer does not contain the keyword '{keyword}'",
            details=details,
        )


@Check.register("regex_matching")
class RegexMatching[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    TextBasedCheck[InputType, OutputType, TraceType]
):
    r"""Check that validates if a regex pattern matches within text.

    Matching uses the PyPI :mod:`regex` package, which is largely compatible
    with Python's standard :mod:`re` module.

    The matching process:
    1. Extracts text and pattern (from provided values or trace)
    2. Searches the text for the pattern

    Attributes
    ----------
    text : str | MISSING
        The text string to search within. If omitted, extracted from
        trace using ``text_key``.
    text_key : JSONPathStr
        JSONPath expression to extract the text from the trace. Defaults to
        "trace.last.outputs" which extracts the last interaction's outputs.
    pattern : str | MISSING
        The regex pattern to search for. Either this or ``pattern_key`` must be provided.
    pattern_key : JSONPathStr | MISSING
        JSONPath expression to extract pattern from trace.
    match_timeout_seconds : float
        Upper bound on how long matching may take before the check raises an error.

    Examples
    --------
    Basic regex matching for prices:

        check = RegexMatching(
            text="Price: $10.99",
            pattern=r"\$\d+\.\d{2}"
        )

    Email validation:

        check = RegexMatching(
            text="Contact: user@example.com",
            pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        )

    Case-insensitive matching (using inline modifier):

        check = RegexMatching(
            text="The ANSWER is 42",
            pattern=r"(?i)answer.*\d+"
        )


    Multiline matching with anchors (using inline modifier):

        check = RegexMatching(
            text="Line 1\nLine 2\nLine 3",
            pattern=r"(?m)^Line 2$"
        )

    Extract from trace:

        check = RegexMatching(
            text_key="trace.last.outputs.response",
            pattern_key="trace.last.inputs.expected_pattern"
        )

    Phone number matching::

        check = RegexMatching(
            text="Call me at 555-123-4567",
            pattern=r"\d{3}-\d{3}-\d{4}"
        )
    """

    pattern: str | MISSING = Field(
        default=MISSING,
        description="The regex pattern to search for within the text.",
    )
    pattern_key: JSONPathStr | MISSING = Field(
        default=MISSING,
        description="JSONPath expression to extract the pattern from the trace.",
    )
    match_timeout_seconds: float = Field(
        default=2.0,
        gt=0,
        description="Maximum time allowed for matching, in seconds.",
    )

    @model_validator(mode="after")
    def validate_pattern_or_pattern_key(self) -> Self:
        """Validate that exactly one of pattern or pattern_key is provided.

        Returns
        -------
        Self
            The validated instance.

        Raises
        ------
        ValueError
            If neither or both pattern and pattern_key are provided.
        """
        if (self.pattern is MISSING) == (self.pattern_key is MISSING):
            raise ValueError(
                "Exactly one of 'pattern' or 'pattern_key' must be provided, not both or neither."
            )
        return self

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the regex matching check.

        Parameters
        ----------
        trace : TraceType
            The trace containing interaction history. Used to extract text/pattern
            if not provided directly.

        Returns
        -------
        CheckResult
            Success if pattern matches text, failure if it doesn't or if regex is invalid.
            Includes details about the text and pattern.
        """
        # Extract and validate using base class
        result = self._extract_and_validate(
            trace, self.pattern, self.pattern_key, "pattern"
        )

        # Check if extraction failed
        if result[0] is None:
            # Error occurred during extraction, return the error result
            return result[2]  # type: ignore[return-value]

        # Extract successful values
        text, pattern, details = result[0], result[1], result[2]

        try:
            matched = regex.search(pattern, text, timeout=self.match_timeout_seconds)
        except regex.error as e:
            return CheckResult.failure(
                message=f"Invalid regex pattern '{pattern}': {str(e)}",
                details=details,
            )

        if matched:
            return CheckResult.success(
                message=f"Text matches the regex pattern '{pattern}'.",
                details=details,
            )
        return CheckResult.failure(
            message=f"Text does not match the regex pattern '{pattern}'.",
            details=details,
        )


class ListTextBasedCheck[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType], ABC
):
    """Base class for checks that validate text against a list of target values.

    This abstract class handles the common pattern of:
    1. Extracting text from a trace or direct value
    2. Extracting a target list of values from trace or direct value
    3. Validating both exist and have correct types
    4. Delegating to subclass for specific matching logic

    Attributes
    ----------
    text : str | MISSING
        The text string to search within. If omitted, extracted from
        trace using ``text_key``.
    text_key : JSONPathStr
        JSONPath expression to extract the text from the trace. Defaults to
        "trace.last.outputs" which extracts the last interaction's outputs.
    values : list[str] | MISSING
        The list of values to search for. Either this or ``values_key`` must be provided.
    values_key : JSONPathStr | MISSING
        JSONPath expression to extract the values from the trace. Either this or ``values`` must be provided.
    normalization_form : NormalizationForm | None
        Unicode normalization form to apply before matching. Options:
        - "NFC": Canonical Composition
        - "NFD": Canonical Decomposition
        - "NFKC": Compatibility Composition (default)
        - "NFKD": Compatibility Decomposition
        If None, no normalization is applied. Defaults to "NFKC".
    case_sensitive : bool
        If True, matching is case-sensitive. If False, both text and values
        are converted to lowercase before comparison. Defaults to False.
    """

    text: str | MISSING = Field(
        default=MISSING,
        description="The text string to search within.",
    )
    text_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract text from trace.",
    )
    values: list[str] | MISSING = Field(
        default=MISSING,
        description="The list of values to search for.",
    )
    values_key: JSONPathStr | MISSING = Field(
        default=MISSING,
        description="JSONPath expression to extract the values from the trace.",
    )
    normalization_form: NormalizationForm | None = Field(
        default="NFKC",
        description="Unicode normalization form to apply (NFC, NFD, NFKC, NFKD). Defaults to NFKC.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="If True, matching is case-sensitive. If False, strings are lowercased before comparison.",
    )

    @model_validator(mode="after")
    def validate_values_or_values_key(self) -> Self:
        """Validate that exactly one of values or values_key is provided.

        Returns
        -------
        Self
            The validated instance.

        Raises
        ------
        ValueError
            If neither or both values and values_key are provided.
        """
        if (self.values is MISSING) == (self.values_key is MISSING):
            raise ValueError(
                "Exactly one of 'values' or 'values_key' must be provided, not both or neither."
            )
        return self

    def _extract_and_validate(
        self,
        trace: TraceType,
    ) -> tuple[str, list[str], dict[str, Any]] | tuple[None, None, CheckResult]:
        """Extract and validate text and target values from trace or direct values.

        Parameters
        ----------
        trace : TraceType
            The trace to extract values from.

        Returns
        -------
        tuple[str, list[str], dict] | tuple[None, None, CheckResult]
            Either (text, values, details) on success, or (None, None, error_result) on failure.
        """
        text = provided_or_resolve(trace, key=self.text_key, value=self.text)
        target = provided_or_resolve(
            trace,
            key=self.values_key,
            value=self.values,
        )

        details = {"text": text, "values": target}

        if isinstance(target, NoMatch):
            return (
                None,
                None,
                CheckResult.failure(
                    message=f"No value found for values key '{self.values_key}'.",
                    details=details,
                ),
            )

        if not isinstance(target, list) or not all(isinstance(x, str) for x in target):
            return (
                None,
                None,
                CheckResult.failure(
                    message="Value for values must be a list of strings.",
                    details=details,
                ),
            )

        if isinstance(text, NoMatch):
            return (
                None,
                None,
                CheckResult.failure(
                    message=f"No value found for text key '{self.text_key}'.",
                    details=details,
                ),
            )

        if not isinstance(text, str):
            return (
                None,
                None,
                CheckResult.failure(
                    message=f"Value for text is not a string, expected string but got {type(text).__name__}.",
                    details=details,
                ),
            )

        return text, target, details

    def _format_str(self, value: str) -> str:
        """Format a string for matching by applying normalization and case handling.

        This method:
        1. Applies Unicode normalization if specified
        2. Converts to lowercase if case-insensitive matching is enabled

        Parameters
        ----------
        value : str
            The string to format.

        Returns
        -------
        str
            The formatted string ready for comparison.
        """
        value = normalize_string(value, self.normalization_form)
        if not self.case_sensitive:
            value = value.lower()
        return value


@Check.register("contains_any")
class ContainsAny[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    ListTextBasedCheck[InputType, OutputType, TraceType]
):
    """Check that validates if at least one value appears within a text string.

    This check performs substring matching between a list of values and text, with
    support for Unicode normalization and case sensitivity control. Both the
    text and values can be provided directly or extracted from a trace using
    JSONPath expressions.

    Attributes
    ----------
    text : str | MISSING
        The text string to search within. If omitted, extracted from
        trace using ``text_key``.
    text_key : JSONPathStr
        JSONPath expression to extract the text from the trace. Defaults to
        "trace.last.outputs" which extracts the last interaction's outputs.
    values : list[str] | MISSING
        The list of values to search for. Either this or ``values_key`` must be provided.
    values_key : JSONPathStr | MISSING
        JSONPath expression to extract the values from the trace. Either this or ``values`` must be provided.
    normalization_form : NormalizationForm | None
        Unicode normalization form to apply before matching. Options:
        - "NFC": Canonical Composition
        - "NFD": Canonical Decomposition
        - "NFKC": Compatibility Composition (default)
        - "NFKD": Compatibility Decomposition
        If None, no normalization is applied. Defaults to "NFKC".
    case_sensitive : bool
        If True, matching is case-sensitive. If False, both text and values
        are converted to lowercase before comparison. Defaults to False.

    Examples
    --------
    Direct text and values::

        check = ContainsAny(
            text="Hello World",
            values=["world", "java"],
            case_sensitive=False
        )

    Extract both from trace::

        check = ContainsAny(
            text_key="trace.last.outputs.answer",
            values_key="trace.last.inputs.expected_keywords"
        )
    """

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the ContainsAny check.

        Parameters
        ----------
        trace : TraceType
            The trace to evaluate.

        Returns
        -------
        CheckResult
            The result of the check execution.
        """
        result = self._extract_and_validate(trace)
        if result[0] is None:
            return result[2]  # type: ignore[return-value]

        text, values, details = result[0], result[1], result[2]
        details["normalization_form"] = self.normalization_form
        details["case_sensitive"] = self.case_sensitive

        formatted_text = self._format_str(text)
        formatted_values = [self._format_str(v) for v in values]

        matched_values = [
            v for v, fv in zip(values, formatted_values) if fv in formatted_text
        ]

        details["matched_values"] = matched_values

        if matched_values:
            return CheckResult.success(
                message=f"The text contains at least one of the values: {', '.join(matched_values)}.",
                details=details,
            )

        return CheckResult.failure(
            message="The text does not contain any of the provided values.",
            details=details,
        )


@Check.register("contains_all")
class ContainsAll[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    ListTextBasedCheck[InputType, OutputType, TraceType]
):
    """Check that validates if all values appear within a text string.

    This check performs substring matching between a list of values and text, with
    support for Unicode normalization and case sensitivity control. Both the
    text and values can be provided directly or extracted from a trace using
    JSONPath expressions.

    Attributes
    ----------
    text : str | MISSING
        The text string to search within. If omitted, extracted from
        trace using ``text_key``.
    text_key : JSONPathStr
        JSONPath expression to extract the text from the trace. Defaults to
        "trace.last.outputs" which extracts the last interaction's outputs.
    values : list[str] | MISSING
        The list of values to search for. Either this or ``values_key`` must be provided.
    values_key : JSONPathStr | MISSING
        JSONPath expression to extract the values from the trace. Either this or ``values`` must be provided.
    normalization_form : NormalizationForm | None
        Unicode normalization form to apply before matching. Options:
        - "NFC": Canonical Composition
        - "NFD": Canonical Decomposition
        - "NFKC": Compatibility Composition (default)
        - "NFKD": Compatibility Decomposition
        If None, no normalization is applied. Defaults to "NFKC".
    case_sensitive : bool
        If True, matching is case-sensitive. If False, both text and values
        are converted to lowercase before comparison. Defaults to False.

    Examples
    --------
    Direct text and values::

        check = ContainsAll(
            text="Hello World",
            values=["hello", "world"],
            case_sensitive=False
        )

    Extract both from trace::

        check = ContainsAll(
            text_key="trace.last.outputs.answer",
            values_key="trace.last.inputs.all_keywords"
        )
    """

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the ContainsAll check.

        Parameters
        ----------
        trace : TraceType
            The trace to evaluate.

        Returns
        -------
        CheckResult
            The result of the check execution.
        """
        result = self._extract_and_validate(trace)
        if result[0] is None:
            return result[2]  # type: ignore[return-value]

        text, values, details = result[0], result[1], result[2]
        details["normalization_form"] = self.normalization_form
        details["case_sensitive"] = self.case_sensitive

        formatted_text = self._format_str(text)
        formatted_values = [self._format_str(v) for v in values]

        missing_values = [
            v for v, fv in zip(values, formatted_values) if fv not in formatted_text
        ]

        details["missing_values"] = missing_values

        if not missing_values:
            return CheckResult.success(
                message="The text contains all of the provided values.",
                details=details,
            )

        return CheckResult.failure(
            message=f"The text is missing the following values: {', '.join(missing_values)}.",
            details=details,
        )
