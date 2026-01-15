from typing import override

from giskard.agents.workflow import TemplateReference
from jinja2 import Template
from pydantic import Field

from ..core.check import Check
from ..core.trace import Trace
from .base import BaseLLMCheck


@Check.register("conformity")
class Conformity[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that validates interactions conform to a given rule.

    This check supports **dynamic rules** by using Jinja2 templating on the `rule`
    string. The entire `Trace` object is exposed to the rule template,
    allowing users to inject trace fields like interactions, inputs, outputs, or metadata.

    Uses an LLM to determine if an interaction (inputs, outputs, and metadata)
    conforms to a specified rule or requirement.

    Attributes
    ----------
    rule : str
        The rule statement to evaluate against the interaction.
        This string can contain Jinja2 placeholders (e.g., `{{ trace.last.outputs }}`).

        Note: In Jinja2 templates and JSONPath expressions, prefer `trace.last` over
        `trace.interactions[-1]` for better readability.
    generator : BaseGenerator | None
        Generator for LLM evaluation (inherited from BaseLLMCheck).

    Examples
    --------
    >>> from giskard.agents.generators import Generator
    >>> from giskard.checks import Interaction, Trace, Conformity
    >>> # Example of a dynamic rule accessing a field in the output object
    >>> check = Conformity(
    ...     rule="The response should contain the keywords '{{ trace.last.inputs.keywords }}' and be polite.",
    ...     generator=Generator(model="openai/gpt-4o")
    ... )
    """

    rule: str = Field(
        ..., description="The rule statement to evaluate against the interaction"
    )

    @override
    def get_prompt(self) -> TemplateReference:
        """Return the Jinja2 template name for conformity evaluation."""
        return TemplateReference(template_name="giskard.checks::checks/conformity.j2")

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, str]:
        """Build template variables from the trace."""
        formatted_rule = Template(self.rule).render(trace=trace)

        interaction_json = "{}"
        if trace.interactions:
            interaction_json = trace.interactions[-1].model_dump_json()

        return {
            "rule": formatted_rule,
            "interaction": interaction_json,
        }
