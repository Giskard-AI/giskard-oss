from counterpoint.chat import Message
from counterpoint.templates import MessageTemplate
from counterpoint.workflow import TemplateReference
from pydantic import Field, model_validator

from giskard_checks.checks.base import BaseLLMCheck
from giskard_checks.core.check import Check


@Check.register("llm_judge")
class LLMJudge(BaseLLMCheck):
    """LLM check that can use either an inline prompt or a prompt file path."""

    prompt: str | None = Field(
        default=None, description="Inline prompt content for the LLM check"
    )
    prompt_path: str | None = Field(
        default=None, description="Path to a file containing the prompt template"
    )

    def get_prompt(self) -> str | Message | MessageTemplate | TemplateReference:
        if self.prompt is not None:
            return self.prompt

        if self.prompt_path is not None:
            return TemplateReference(template_name=self.prompt_path)

        raise ValueError("Either 'prompt' or 'prompt_path' must be provided")

    @model_validator(mode="after")
    def validate_prompt_or_path(self) -> "LLMJudge":
        """Validate that exactly one of prompt or prompt_path is provided."""
        if self.prompt is None and self.prompt_path is None:
            raise ValueError("Either 'prompt' or 'prompt_path' must be provided")
        if self.prompt is not None and self.prompt_path is not None:
            raise ValueError(
                "Cannot provide both 'prompt' and 'prompt_path' - choose one"
            )
        return self
