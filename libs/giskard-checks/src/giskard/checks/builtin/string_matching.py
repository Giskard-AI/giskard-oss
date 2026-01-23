"""String matching check implementation.

This module provides a check that validates whether a keyword appears within
a text string. It supports Unicode normalization, case sensitivity control,
and flexible text/keyword extraction from traces.
"""

from __future__ import annotations

from typing import Self, override

from giskard.core import provide_not_none
from pydantic import Field, model_validator

from ..core.check import Check
from ..core.extraction import NoMatch, provided_or_resolve
from ..core.result import CheckResult
from ..core.trace import Trace
from ..utils.normalization import NormalizationForm, normalize_string


@Check.register("string_matching")
class StringMatching[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
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
    text : str | None
        The text string to search within. If None, will be extracted from
        trace using `text_key`.
    text_key : str
        JSONPath expression to extract the text from the trace. Defaults to
        "trace.last.outputs" which extracts the last interaction's outputs.
    keyword : str | None
        The keyword to search for within the text. If None, must provide
        `keyword_key` to extract from trace.
    keyword_key : str | None
        JSONPath expression to extract the keyword from the trace. Either
        `keyword` or `keyword_key` must be provided.
    normalization_form : _NormalizationForm | None
        Unicode normalization form to apply before matching. Options:
        - "NFC": Canonical Composition (default)
        - "NFD": Canonical Decomposition
        - "NFKC": Compatibility Composition
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

    text: str | None = Field(
        default=None,
        description="The text string to search within. If None, extracted from trace using text_key.",
    )
    text_key: str = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the text from the trace (e.g., 'trace.last.outputs.response').",
    )
    keyword: str | None = Field(
        default=None,
        description="The keyword to search for within the text. Either this or keyword_key must be provided.",
    )
    keyword_key: str | None = Field(
        default=None,
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
            If neither keyword nor keyword_key is provided.
        """
        if self.keyword is None and self.keyword_key is None:
            raise ValueError("Either 'keyword' or 'keyword_key' must be provided")

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
        text = provided_or_resolve(
            trace, key=self.text_key, value=provide_not_none(self.text)
        )
        keyword = provided_or_resolve(
            trace,
            key=provide_not_none(self.keyword_key),
            value=provide_not_none(self.keyword),
        )

        details = {
            "text": text,
            "keyword": keyword,
            "normalization_form": self.normalization_form,
            "case_sensitive": self.case_sensitive,
        }

        if isinstance(keyword, NoMatch):
            return CheckResult.failure(
                message=f"No value found for keyword key '{self.keyword_key}'.",
                details=details,
            )

        if not isinstance(keyword, str):
            return CheckResult.failure(
                message=f"Value for keyword key '{self.keyword_key}' is not a string, expected string but got {type(keyword)}.",
                details=details,
            )

        if isinstance(text, NoMatch):
            return CheckResult.failure(
                message=f"No value found for text key '{self.text_key}', expected string to contain '{keyword}'.",
                details=details,
            )

        if not isinstance(text, str):
            return CheckResult.failure(
                message=f"Value for text key '{self.text_key}' is not a string, expected string to contain '{keyword}' but got {type(text)}.",
                details=details,
            )

        # Format both strings for comparison
        formatted_text = self._format_str(str(text))
        formatted_keyword = self._format_str(str(keyword))

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
