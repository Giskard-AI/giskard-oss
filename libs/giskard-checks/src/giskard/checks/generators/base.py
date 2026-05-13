from collections.abc import AsyncGenerator
from typing import Any, Self, override

from giskard.agents.templates import MessageTemplate
from giskard.agents.workflow import ChatWorkflow, TemplateReference
from giskard.llm import chat
from giskard.llm.types import ChatMessage
from pydantic import BaseModel, Field, model_validator

from ..core import Trace
from ..core.exceptions import InputGenerationException
from ..core.input_generator import InputGenerator
from ..core.mixin import WithGeneratorMixin


class LLMGeneratorOutput[T: str | BaseModel](BaseModel):
    goal_reached: bool = Field(
        ...,
        description="Whether the goal has been reached and no more messages are needed.",
    )
    schema_issue: str | None = Field(
        default=None,
        description="Schema issue preventing message generation (e.g. no string-like field). "
        "Set this instead of message when the schema cannot produce a user message.",
    )
    message: T | None = Field(
        default=None,
        description="The message to send. None if goal_reached is True.",
    )

    @model_validator(mode="after")
    def _validate_message_and_schema_issue(self) -> "LLMGeneratorOutput[T]":
        if self.message is not None and self.schema_issue is not None:
            raise ValueError("'message' and 'schema_issue' cannot both be set")
        return self


class BaseLLMGenerator[TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    InputGenerator[TraceType], WithGeneratorMixin
):
    """Abstract base for LLM-driven multi-turn input generators.

    Mirrors BaseLLMCheck on the generator side. Subclasses implement
    get_prompt() and optionally override get_inputs().
    """

    max_steps: int = Field(default=3, ge=0)

    def get_prompt(self) -> ChatMessage | TemplateReference | MessageTemplate:
        """Return the prompt. Subclasses must override."""
        raise NotImplementedError

    async def get_inputs(self, trace: TraceType) -> dict[str, Any]:
        """Return template variables. Default provides trace only."""
        return {"trace": trace}

    @override
    async def __call__(
        self, trace: TraceType, input_type: type[str | BaseModel] | None = None
    ) -> AsyncGenerator[Any, TraceType]:
        T = input_type or str
        prompt = self.get_prompt()

        if isinstance(prompt, TemplateReference):
            workflow = self.generator.template(prompt.template_name).with_output(
                LLMGeneratorOutput[T]
            )
        else:
            workflow = ChatWorkflow(
                generator=self.generator,
                messages=[prompt],
            ).with_output(LLMGeneratorOutput[T])

        step = 0
        while step < self.max_steps:
            inputs = await self.get_inputs(trace)
            result = await workflow.with_inputs(**inputs).run()
            output = result.output

            if output.schema_issue:
                raise InputGenerationException(f"schema issue: {output.schema_issue}")
            if output.goal_reached or not output.message:
                return

            trace = yield output.message
            step += 1


@InputGenerator.register("llm_generator")
class LLMGenerator[TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMGenerator[TraceType]
):
    """Configurable LLM-driven generator with inline prompt or template path.

    Mirrors LLMJudge on the generator side. Exactly one of prompt or
    prompt_path must be provided.

    Parameters
    ----------
    prompt : str | None
        Inline prompt string. Jinja2 rendering only applies when `as_template=True`.
    prompt_path : str | None
        Template reference (e.g. "giskard.checks::scenarios/my_template.j2").
    max_steps : int
        Maximum conversation turns (default: 3).

    Examples
    --------
    >>> gen = LLMGenerator(prompt="You are a user. Ask about the product.")
    >>> gen = LLMGenerator(prompt_path="giskard.checks::scenarios/llm01.j2")
    """

    prompt: str | None = Field(default=None, description="Inline prompt string.")
    as_template: bool = Field(
        default=False,
        description="Whether to render the prompt as jinja2 template. prompt_path is always rendered as a template.",
    )
    prompt_path: str | None = Field(
        default=None, description="Template file reference."
    )

    @model_validator(mode="after")
    def _validate_prompt_xor_path(self) -> Self:
        if self.prompt is None and self.prompt_path is None:
            raise ValueError("Either 'prompt' or 'prompt_path' must be provided")
        if self.prompt is not None and self.prompt_path is not None:
            raise ValueError(
                "Cannot provide both 'prompt' and 'prompt_path' - choose one"
            )
        return self

    @override
    def get_prompt(self) -> ChatMessage | TemplateReference | MessageTemplate:
        if self.prompt is not None:
            return (
                MessageTemplate(role="user", content_template=self.prompt)
                if self.as_template
                else chat.user(self.prompt)
            )

        assert self.prompt_path is not None
        return TemplateReference(template_name=self.prompt_path)
