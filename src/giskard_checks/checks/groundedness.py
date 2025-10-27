from typing import Any

from pydantic import Field

from giskard_checks.checks.llm_check import TemplateLLMCheck
from giskard_checks.core.check import Check
from giskard_checks.core.extraction import resolve
from giskard_checks.core.interaction_result import InteractionResult


@Check.register("groundedness")
class Groundedness(TemplateLLMCheck):
    """LLM-based check that validates answers are grounded in context.

    Uses an LLM to determine if an answer is properly supported by
    the provided context documents.

    Attributes
    ----------
    answer : str | None
        The answer text to evaluate for groundedness.
    answer_key : str
        Key to extract the answer from the interaction (default: "$.outputs").
    context : list[str] | None
        List of context documents that should support the answer.
    context_key : str
        Key to extract the context from the interaction (default: "$.metadata.context").
    generator : BaseGenerator | None
        Counterpoint generator for LLM evaluation (inherited from LLMCheck).

    Examples
    --------
    >>> import counterpoint as cp
    >>> check = Groundedness(
    ...     answer="The Eiffel Tower is in Paris.",
    ...     context=["Paris is the capital of France.", "It's located in Europe."],
    ...     generator=cp.Generator(model="openai/gpt-4o")
    ... )
    """

    answer: str | None = Field(
        default=None, description="Input source for the answer to evaluate"
    )
    answer_key: str = Field(
        default="$.outputs",
        description="Key to extract the answer from the interaction",
    )
    context: list[str] | None = Field(
        default=None, description="Input source for the reference context"
    )
    context_key: str = Field(
        default="$.metadata.context",
        description="Key to extract the context from the interaction",
    )

    @property
    def template_name(self) -> str:
        """Return the Jinja2 template name for groundedness evaluation."""
        return "giskard_checks::checks/groundedness.j2"

    async def _build_template_inputs(
        self, interaction: InteractionResult[Any, Any]
    ) -> dict[str, str]:
        """Build template variables from resolved inputs.

        Parameters
        ----------
        interaction : Interaction[Any, Any]
            Context for resolving inputs.

        Returns
        -------
        dict[str, str]
            Template variables with 'answer' and 'context' keys.
        """
        return {
            "answer": self.answer or str(resolve(interaction, self.answer_key)),
            "context": str(
                self.context or resolve(interaction, self.context_key, multiple=True)
            ),
        }
