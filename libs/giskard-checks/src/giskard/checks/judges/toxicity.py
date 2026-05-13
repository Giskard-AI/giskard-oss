from typing import Any, Literal, override

from giskard.agents.workflow import TemplateReference
from giskard.core import provide_not_none
from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, provided_or_resolve
from .base import BaseLLMCheck

ToxicityCategory = Literal[
    "hate_speech",
    "harassment",
    "threats",
    "self_harm",
    "sexual_content",
    "violence",
]

DEFAULT_TOXICITY_CATEGORIES: tuple[ToxicityCategory, ...] = (
    "hate_speech",
    "harassment",
    "threats",
    "self_harm",
    "sexual_content",
    "violence",
)


@Check.register("toxicity")
class Toxicity[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that detects toxic, harmful, or offensive content.

    Uses an LLM judge to determine whether the model output contains toxic
    content across configurable categories including hate speech, harassment,
    threats, self-harm, sexual content, and violence.

    Attributes
    ----------
    output : str | None
        The text to evaluate for toxicity. If None, extracted from the trace
        using ``output_key``.
    output_key : JSONPathStr
        JSONPath expression to extract the output from the trace
        (default: ``"trace.last.outputs"``).

        Can use ``trace.last`` (preferred) or ``trace.interactions[-1]`` for
        JSONPath expressions.
    categories : list[ToxicityCategory]
        Specific toxicity categories to evaluate. Defaults to all built-in
        categories: ``hate_speech``, ``harassment``, ``threats``, ``self_harm``,
        ``sexual_content``, ``violence``. Providing an explicit list restricts
        the judge to only those categories.
    generator : BaseGenerator | None
        Generator for LLM evaluation (inherited from BaseLLMCheck).

    Examples
    --------
    Check for all toxicity categories using a trace:

    >>> from giskard.checks import Toxicity, Scenario
    >>> scenario = (
    ...     Scenario(name="safety_check")
    ...     .interact(inputs="Tell me a joke", outputs="Here is a clean joke: ...")
    ...     .check(Toxicity())
    ... )

    Check only for hate speech and harassment:

    >>> from giskard.agents.generators import Generator
    >>> check = Toxicity(
    ...     output="This is a safe response.",
    ...     categories=["hate_speech", "harassment"],
    ...     generator=Generator(model="openai/gpt-4o"),
    ... )
    """

    output: str | None = Field(
        default=None,
        description="The text to evaluate for toxicity. If None, extracted from the trace using output_key.",
    )
    output_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the output from the trace.",
    )
    categories: list[ToxicityCategory] = Field(
        default_factory=lambda: list(DEFAULT_TOXICITY_CATEGORIES),
        description=(
            "Specific toxicity categories to evaluate. "
            "Defaults to all built-in categories: "
            "hate_speech, harassment, threats, self_harm, sexual_content, violence."
        ),
    )

    @override
    def get_prompt(self) -> TemplateReference:
        """Return the bundled prompt template for toxicity evaluation."""
        return TemplateReference(template_name="giskard.checks::judges/toxicity.j2")

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, Any]:
        """Build template variables for the toxicity judge prompt.

        Parameters
        ----------
        trace : Trace
            Trace for resolving inputs.

        Returns
        -------
        dict[str, Any]
            Template variables with ``output``, ``categories``, and ``trace``
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
        }
