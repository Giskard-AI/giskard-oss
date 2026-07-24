"""Bias check implementation.

This module provides an LLM-based check for detecting biased content in AI
agent responses, including stereotyping, discrimination, and unfair
representation across demographic groups.
"""

from typing import Any, override

from giskard.core import provide_not_none
from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, provided_or_resolve
from .base import BaseLLMCheck

# Default protected attributes derived from:
# - EU AI Act, Article 5 & Annex III (prohibited discrimination grounds):
#   https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689
# - DeepEval BiasMetric (gender, religion, race, politics):
#   https://docs.confident-ai.com/docs/metrics-bias
# - ISO/IEC 24368:2022 (AI fairness standard referencing demographic attributes)
DEFAULT_PROTECTED_ATTRIBUTES: list[str] = [
    "gender",
    "race",
    "age",
    "religion",
    "nationality",
    "sexual_orientation",
    "socioeconomic_status",
    "disability",
]


@Check.register("bias")
class Bias[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that detects biased content in AI agent responses.

    Uses an LLM judge to detect stereotyping, discrimination, and unfair
    representation across configurable demographic dimensions such as gender,
    race, age, or religion.

    Attributes
    ----------
    output : str | None
        The text to evaluate for bias. If ``None``, extracted from the trace
        using ``key``.
    key : JSONPathStr
        JSONPath expression to extract the output to evaluate from the trace
        (default: ``"trace.last.outputs"``).
    protected_attributes : list[str] | None
        Specific demographic attributes to check for bias (e.g.
        ``["gender", "race", "age"]``). If ``None``, all default attributes
        are evaluated: gender, race, age, religion, nationality,
        sexual_orientation, socioeconomic_status, disability.
    context_key : JSONPathStr | None
        JSONPath expression to extract context/input from the trace for
        evaluating relative bias (e.g. to detect when the output endorses
        a biased premise in the input). If ``None``, bias is evaluated on
        the output alone.
    attribute_descriptions : dict[str, str] | None
        Per-attribute description overrides forwarded to the prompt. Keys must
        match entries in ``protected_attributes``. Unspecified attributes use
        the default prompt description. Example::

            {"gender": "Look for assumptions about professional roles based on gender"}

    generator : BaseGenerator | None
        Generator for LLM evaluation (inherited from BaseLLMCheck).

    Examples
    --------
    Check for gender and racial bias using a trace:

    >>> from giskard.checks import Bias, Scenario
    >>> scenario = (
    ...     Scenario(name="bias_check")
    ...     .interact(inputs="Describe a software engineer", outputs="...")
    ...     .check(Bias(protected_attributes=["gender", "race"]))
    ... )

    Check with a direct output string:

    >>> check = Bias(
    ...     output="Women tend to be more nurturing.",
    ...     protected_attributes=["gender"],
    ... )

    Check with context for relative bias evaluation:

    >>> from giskard.agents.generators import Generator
    >>> check = Bias(
    ...     protected_attributes=["gender"],
    ...     context_key="trace.last.inputs",
    ...     generator=Generator(model="openai/gpt-4o"),
    ... )
    """

    output: str | None = Field(
        default=None,
        description="The text to evaluate for bias. If None, extracted from the trace using key.",
    )
    key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the output to evaluate from the trace.",
    )
    protected_attributes: list[str] | None = Field(
        default=None,
        description=(
            "Specific demographic attributes to check for bias "
            "(e.g. ['gender', 'race', 'age']). "
            "If None, all default attributes are evaluated."
        ),
    )
    context_key: JSONPathStr | None = Field(
        default=None,
        description=(
            "JSONPath expression to extract context/input from the trace for "
            "evaluating relative bias. If None, bias is evaluated on output alone."
        ),
    )
    attribute_descriptions: dict[str, str] | None = Field(
        default=None,
        description=(
            "Per-attribute description overrides for the prompt. "
            "Keys must match entries in protected_attributes. "
            'Example: {"gender": "Look for assumptions about professional roles based on gender"}. '
            "Unspecified attributes use the default prompt description."
        ),
    )

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, Any]:
        """Build template variables for the bias judge prompt.

        Parameters
        ----------
        trace : Trace
        Trace for resolving inputs.

        Returns
        -------
        dict[str, Any]
        Template variables with ``output``, ``protected_attributes``,
        ``context``, and ``trace`` keys.
        """
        attributes = (
            self.protected_attributes
            if self.protected_attributes is not None
            else DEFAULT_PROTECTED_ATTRIBUTES
        )

        # Resolve context if context_key is provided
        context: str | None = None
        if self.context_key is not None:
            resolved = provided_or_resolve(
                trace, key=self.context_key, value=provide_not_none(None)
            )
            if not isinstance(resolved, NoMatch) and resolved is not None:
                context = str(resolved)

        # Resolve output
        resolved_output = provided_or_resolve(
            trace,
            key=self.key,
            value=provide_not_none(self.output),
        )
        if isinstance(resolved_output, NoMatch) or resolved_output is None:
            raise ValueError(
                f"Could not resolve output for bias check using key '{self.key}'"
            )

        return {
            "trace": trace,
            "output": str(resolved_output),
            "protected_attributes": attributes,
            "attribute_descriptions": self.attribute_descriptions or {},
            "context": context,
        }
