from typing import Any

from giskard.agents.workflow import TemplateReference
from jinja2 import Template
from pydantic import BaseModel, Field

from ..checks.base import BaseLLMCheck
from ..core.check import Check
from ..core.interaction_result import InteractionResult


@Check.register("conformity")
class Conformity(BaseLLMCheck):
    """LLM-based check that validates interactions conform to a given rule.

    This check supports **dynamic rules** by using Jinja2 templating on the `rule`
    string. The entire `Interaction` object is exposed to the rule template,
    allowing users to inject interaction fields like inputs, outputs, or metadata.

    Uses an LLM to determine if an interaction (inputs, outputs, and metadata)
    conforms to a specified rule or requirement.

    Attributes
    ----------
    rule : str
        The rule statement to evaluate against the interaction.
        This string can contain Jinja2 placeholders (e.g., `{{ outputs.text_value }}`).
    generator : BaseGenerator | None
        Generator for LLM evaluation (inherited from BaseLLMCheck).

    Examples
    --------
    >>> from giskard.agents.generators import Generator
    >>> from giskard.checks.core.interaction_result import InteractionResult
    >>> # Example of a dynamic rule accessing a field in the output object
    >>> check = Conformity(
    ...     rule="The response should contain the keywords '{{ inputs.keywords }}' and be polite.",
    ...     generator=Generator(model="openai/gpt-4o")
    ... )
    >>>
    >>> # The 'rule' string is rendered using all fields of the Interaction model.
    >>> test_case = TestCase(
    ...     name="test_adversarial_generation",
    ...     interaction=interaction,
    ...     checks=[
    ...         Conformity("The inputs should be relevant to the adversarial category."),
    ...         Conformity("The outputs must not mention '{{ inputs.sensitive_word }}'."),
    ...     ]
    ... )
    """

    rule: str = Field(
        ..., description="The rule statement to evaluate against the interaction"
    )

    def get_prompt(self) -> TemplateReference:
        """Return the Jinja2 template name for conformity evaluation."""
        return TemplateReference(template_name="giskard.checks::checks/conformity.j2")

    def _format_data(self, data: Any) -> str:
        """Formats data for the LLM prompt template."""
        if isinstance(data, BaseModel):
            # Use .model_dump_json() for a cleaner, universal representation
            return data.model_dump_json(indent=2)
        # Use repr() as a fallback for non-Pydantic types
        return repr(data)

    async def get_inputs(
        self, interaction: InteractionResult[Any, Any]
    ) -> dict[str, str]:
        """Build template variables from the interaction."""

        formatted_rule = Template(self.rule).render(**interaction.model_dump())

        interaction_inputs = self._format_data(interaction.inputs)
        interaction_outputs = self._format_data(interaction.outputs)
        interaction_metadata = self._format_data(interaction.metadata)

        return {
            "rule": formatted_rule,
            "interaction_inputs": interaction_inputs,
            "interaction_outputs": interaction_outputs,
            "interaction_metadata": interaction_metadata,
        }
