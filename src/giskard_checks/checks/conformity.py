from typing import Any

from jinja2 import Template
from pydantic import BaseModel, Field

from giskard_checks.checks.llm_check import TemplateLLMCheck
from giskard_checks.core.check import Check
from giskard_checks.core.interaction_result import InteractionResult


@Check.register("conformity")
class Conformity(TemplateLLMCheck):
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
        Counterpoint generator for LLM evaluation (inherited from LLMCheck).

    Examples
    --------
    >>> import counterpoint as cp
    >>> from giskard_checks.core.interaction_result import InteractionResult
    >>> # Example of a dynamic rule accessing a field in the output object
    >>> check = Conformity(
    ...     rule="The response should contain the keywords '{{ inputs.keywords }}' and be polite.",
    ...     generator=cp.Generator(model="openai/gpt-4o")
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

    @property
    def template_name(self) -> str:
        """Return the Jinja2 template name for conformity evaluation."""
        return "giskard_checks::checks/conformity.j2"

    def _format_data(self, data: Any) -> str:
        """Formats data for the LLM prompt template."""
        if isinstance(data, BaseModel):
            # Use .model_dump_json() for a cleaner, universal representation
            return data.model_dump_json(indent=2)
        # Use repr() as a fallback for non-Pydantic types
        return repr(data)

    async def _build_template_inputs(
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
