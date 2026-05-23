from typing import Any, Literal, override

from giskard.agents.workflow import TemplateReference
from giskard.core import provide_not_none
from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, provided_or_resolve
from .base import BaseLLMCheck

PIICategory = Literal[
    "email",
    "phone",
    "ssn",
    "credit_card",
    "ip_address",
    "name",
    "address",
    "medical",
    "financial",
]

DEFAULT_PII_CATEGORIES: tuple[PIICategory, ...] = (
    "email",
    "phone",
    "ssn",
    "credit_card",
    "ip_address",
    "name",
    "address",
    "medical",
    "financial",
)

PIIDetectionMode = Literal["pattern", "llm", "hybrid"]


@Check.register("pii_detection")
class PIIDetection[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that detects personally identifiable information (PII).

    Uses an LLM judge to determine whether the model output contains personally
    identifiable information across configurable categories including email,
    phone, SSN, credit card, IP address, name, address, medical info, and
    financial info.

    Attributes
    ----------
    output : str | None
        The text to evaluate for PII. If None, extracted from the trace
        using ``output_key``.
    output_key : JSONPathStr
        JSONPath expression to extract the output from the trace
        (default: ``"trace.last.outputs"``).

        Can use ``trace.last`` (preferred) or ``trace.interactions[-1]`` for
        JSONPath expressions.
    categories : list[PIICategory]
        Specific PII categories to evaluate. Defaults to all built-in
        categories: ``email``, ``phone``, ``ssn``, ``credit_card``,
        ``ip_address``, ``name``, ``address``, ``medical``, ``financial``.
        Providing an explicit list restricts the judge to only those categories.
    mode : PIIDetectionMode
        Detection mode to use: ``"pattern"`` (regex-based), ``"llm"`` (LLM-based),
        or ``"hybrid"`` (both). Default is ``"llm"``.
    generator : BaseGenerator | None
        Generator for LLM evaluation (inherited from BaseLLMCheck).

    Examples
    --------
    Check for all PII categories using a trace:

    >>> from giskard.checks import PIIDetection, Scenario
    >>> scenario = (
    ...     Scenario(name="pii_check")
    ...     .interact(inputs="What is your email?", outputs="My email is john@example.com")
    ...     .check(PIIDetection())
    ... )

    Check only for email and phone numbers:

    >>> from giskard.agents.generators import Generator
    >>> check = PIIDetection(
    ...     output="Call me at 555-1234 or email info@example.com",
    ...     categories=["email", "phone"],
    ...     generator=Generator(model="openai/gpt-4o"),
    ... )
    """

    output: str | None = Field(
        default=None,
        description="The text to evaluate for PII. If None, extracted from the trace using output_key.",
    )
    output_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the output from the trace.",
    )
    categories: list[PIICategory] = Field(
        default_factory=lambda: list(DEFAULT_PII_CATEGORIES),
        description=(
            "Specific PII categories to evaluate. "
            "Defaults to all built-in categories: "
            "email, phone, ssn, credit_card, ip_address, name, address, medical, financial."
        ),
    )
    mode: PIIDetectionMode = Field(
        default="llm",
        description=(
            "Detection mode: 'pattern' for regex-based, "
            "'llm' for LLM-based, or 'hybrid' for both."
        ),
    )

    @override
    def get_prompt(self) -> TemplateReference:
        """Return the bundled prompt template for PII detection evaluation."""
        return TemplateReference(template_name="giskard.checks::judges/pii_detection.j2")

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, Any]:
        """Build template variables for the PII detection judge prompt.

        Parameters
        ----------
        trace : Trace
            Trace for resolving inputs.

        Returns
        -------
        dict[str, Any]
            Template variables with ``output``, ``categories``, ``mode``, and ``trace``
            keys. The ``trace`` key is inherited from the base class so that
            custom templates can access interaction history or metadata.
        """
        return {
            "trace": trace,
            "output": str(
                provided_or_resolve(
                    trace,
                    key=self.output_key,
                    value=provide_not_none(self.output),
                )
            ),
            "categories": self.categories,
            "mode": self.mode,
        }
