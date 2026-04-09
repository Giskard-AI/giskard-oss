from typing import Any, override

from giskard.agents.workflow import TemplateReference
from pydantic import Field

from ..core import Trace
from ..core.check import Check
from .base import BaseLLMCheck


@Check.register("conformity")
class Conformity[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that validates a trace against a given rule.

    The `rule` is plain text: it is passed to the bundled prompt as-is (not
    evaluated as its own template). The full `Trace` is supplied separately to
    that prompt so the model can judge outputs and metadata against the rule.

    Uses an LLM to determine whether the trace's outputs and metadata conform
    to the rule.

    Attributes
    ----------
    rule : str
        The rule statement to evaluate against the trace (literal text).

    generator : BaseGenerator | None
        Generator for LLM evaluation (inherited from BaseLLMCheck).

    Examples
    --------
    >>> from giskard.agents.generators import Generator
    >>> from giskard.checks import Conformity
    >>> check = Conformity(
    ...     rule="The last response should be polite.",
    ...     generator=Generator(model="openai/gpt-5-mini")
    ... )
    """

    rule: str = Field(
        ..., description="The rule statement to evaluate against the trace (plain text)"
    )

    @override
    def get_prompt(self) -> TemplateReference:
        """Return the bundled prompt template for conformity evaluation."""
        return TemplateReference(template_name="giskard.checks::judges/conformity.j2")

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, Any]:
        """Build template variables from the trace."""
        return {
            "rule": self.rule,
            "trace": trace,
        }
