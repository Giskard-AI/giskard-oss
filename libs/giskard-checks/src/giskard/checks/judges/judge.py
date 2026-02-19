from typing import Self, override

from giskard.agents.chat import Message
from giskard.agents.templates import MessageTemplate
from giskard.agents.workflow import TemplateReference
from pydantic import Field, model_validator

from ..core.check import Check
from ..core.trace import Trace
from .base import BaseLLMCheck


@Check.register("llm_judge")
class LLMJudge[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that evaluates interactions using a custom prompt.

    This check uses a Large Language Model to evaluate interactions based on
    a user-provided prompt. The prompt can be specified either inline as a string
    or as a path to a template file.

    The prompt supports template interpolation using Jinja2 syntax. Available
    template variables include the trace object (accessible via `trace`) and
    any custom variables provided by overriding `get_inputs()`.

    Common template variables:
    - `trace`: The trace object containing all interactions
    - `trace.interactions`: List of all interactions
    - `trace.last`: Most recent interaction (preferred in prompt templates)
    - `trace.last.inputs`: Inputs from the most recent interaction
    - `trace.last.outputs`: Outputs from the most recent interaction
    - `trace.interactions[-1]`: Alternative way to access the most recent interaction
    - Custom Trace subclasses may expose additional properties (e.g., messages or transcripts)

    Note
    ----
    In Jinja2 prompt templates and JSONPath expressions, prefer `trace.last` over
    `trace.interactions[-1]` for better readability. The `trace.last` computed field
    is available in both contexts.

    The LLM is expected to return a structured output with a `passed` boolean field
    and an optional `reason` string field (see `LLMCheckResult`).

    Attributes
    ----------
    prompt : str | None
        Inline prompt content for the LLM evaluation. Supports Jinja2 template
        syntax with access to the trace and interaction data.
    prompt_path : str | None
        Path to a template file containing the prompt. The path should be
        registered with `giskard-agents` template system (e.g., `"checks::judge.j2"`).

    Examples
    --------
    Inline prompt::

        LLMJudge(
            prompt=(
                "Evaluate if the response is helpful and accurate.\\n"
                "Input: {{ trace.last.inputs }}\\n"
                "Response: {{ trace.last.outputs }}\\n"
                "Return passed=true if helpful, passed=false otherwise."
            )
        )

    Template file::

        LLMJudge(
            prompt_path="checks::safety_check.j2"
        )

    Note
    ----
    Exactly one of `prompt` or `prompt_path` must be provided. Providing both
    will raise a `ValueError`.
    """

    prompt: str | None = Field(
        default=None, description="Inline prompt content for the LLM check"
    )
    prompt_path: str | None = Field(
        default=None, description="Path to a file containing the prompt template"
    )

    @override
    def get_prompt(self) -> str | Message | MessageTemplate | TemplateReference:
        if self.prompt is not None:
            return self.prompt

        if self.prompt_path is not None:
            return TemplateReference(template_name=self.prompt_path)

        raise ValueError("Either 'prompt' or 'prompt_path' must be provided")

    @model_validator(mode="after")
    def validate_prompt_or_path(self) -> Self:
        """Validate that exactly one of prompt or prompt_path is provided."""
        if self.prompt is None and self.prompt_path is None:
            raise ValueError("Either 'prompt' or 'prompt_path' must be provided")
        if self.prompt is not None and self.prompt_path is not None:
            raise ValueError(
                "Cannot provide both 'prompt' and 'prompt_path' - choose one"
            )
        return self
