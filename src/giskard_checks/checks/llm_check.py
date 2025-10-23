from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

from counterpoint.generators import Generator
from counterpoint.generators.base import BaseGenerator
from counterpoint.workflow import ChatWorkflow
from pydantic import BaseModel, Field

from giskard_checks.core.check import Check, CheckResult
from giskard_checks.core.interactions import Interaction


class LLMCheckResult(BaseModel):
    """Default result model for LLM-based checks."""

    passed: bool = Field(..., description="Whether the check passed or failed")
    reason: str | None = Field(
        default=None, description="Optional explanation for the result"
    )


class LLMCheck(Check, ABC):
    """Abstract base class for LLM-powered validation checks.

    Provides infrastructure for checks that use LLMs to evaluate
    interactions. Subclasses must implement _build_chat_workflow and
    _build_template_inputs to define check-specific behavior.

    Attributes
    ----------
    generator : BaseGenerator | None
        Counterpoint generator for LLM evaluation. If None, uses global default or GPT-4o-mini.

    Examples
    --------
    >>> import counterpoint as cp
    >>> @Check.register("sentiment")
    ... class SentimentCheck(TemplateLLMCheck):
    ...     text: InputField[str]
    ...
    ...     @property
    ...     def template_name(self) -> str:
    ...         return "checks/sentiment.j2"
    ...
    ...     async def _build_template_inputs(self, interaction: Interaction) -> dict[str, str]:
    ...         return {"text": resolve(self.text, interaction)}
    ...
    >>> # Use with specific generator
    >>> check = SentimentCheck(text="Hello", generator=cp.Generator(model="openai/gpt-4o"))
    >>> # Or set global default
    >>> giskard_checks.set_default_generator(cp.Generator(model="openai/gpt-4o"))
    >>> check = SentimentCheck(text="Hello")  # Uses global default
    """

    generator: BaseGenerator | None = Field(
        default=None,
        exclude=True,  # Not serializable
        description="Counterpoint generator for LLM evaluation",
    )

    @property
    def output_type(self) -> type[BaseModel]:
        return LLMCheckResult

    @cached_property
    def _generator(self) -> BaseGenerator:
        if self.generator is not None:
            return self.generator

        # Fall back to global default
        from giskard_checks import get_default_generator

        default = get_default_generator()
        if default is None:
            # Provide sensible default
            return Generator(model="openai/gpt-4o-mini")
        return default

    @abstractmethod
    async def _build_chat_workflow(
        self, interaction: Interaction[Any, Any]
    ) -> ChatWorkflow[Any]:
        """Build the ChatWorkflow for this check.

        Subclasses must implement this to create the appropriate
        ChatWorkflow (file-based or inline template).

        Parameters
        ----------
        interaction : Interaction
            The interaction to validate.

        Returns
        -------
        ChatWorkflow
            Configured workflow ready to run
        """
        raise NotImplementedError

    async def run(
        self,
        interaction: Interaction[Any, Any],
    ) -> CheckResult:
        """Execute LLM-based validation using the configured ChatWorkflow.

        Parameters
        ----------
        interaction : Interaction
            The interaction to validate.

        Returns
        -------
        CheckResult
            Result based on LLM evaluation.
        """
        chat_workflow = await self._build_chat_workflow(interaction)
        chat = await chat_workflow.run()

        template_inputs = await self._build_template_inputs(interaction)
        return await self._handle_output(chat.output, template_inputs, interaction)

    @abstractmethod
    async def _build_template_inputs(
        self, interaction: Interaction[Any, Any]
    ) -> dict[str, str]:
        """Build template variables from the interaction.

        Subclasses must implement this to extract values for their
        specific template.

        Parameters
        ----------
        interaction : Interaction
            Context for resolving inputs.

        Returns
        -------
        dict[str, str]
            Template variables as key-value pairs.
        """
        raise NotImplementedError(
            "Subclasses must implement _build_template_inputs to build template variables from context"
        )

    async def _handle_output(
        self,
        output_value: BaseModel,
        template_inputs: dict[str, str],
        interaction: Interaction[Any, Any],
    ) -> CheckResult:
        """Convert LLM output to CheckResult.

        Default implementation handles LLMCheckResult. Override for
        custom output types.

        Parameters
        ----------
        output_value : BaseModel
            The structured output from the LLM.
        template_inputs : dict[str, str]
            The template inputs used for the evaluation.
        interaction : Interaction
            The original interaction.

        Returns
        -------
        CheckResult
            Success or failure based on LLM output.
        """
        if isinstance(output_value, LLMCheckResult):
            if output_value.passed:
                return CheckResult.success(
                    message=output_value.reason or "Check passed",
                    details={
                        "reason": output_value.reason,
                        "inputs": template_inputs,
                    },
                )
            else:
                return CheckResult.failure(
                    message=output_value.reason or "Check failed",
                    details={
                        "reason": output_value.reason,
                        "inputs": template_inputs,
                    },
                )

        raise NotImplementedError(
            f"Custom output type {type(output_value)} requires overriding _handle_output"
        )


class TemplateLLMCheck(LLMCheck, ABC):
    """Abstract base class for LLM checks using file-based Jinja2 templates.

    Subclasses must implement template_name and _build_template_inputs.

    Examples
    --------
    >>> @Check.register("sentiment")
    ... class SentimentCheck(TemplateLLMCheck):
    ...     text: str
    ...
    ...     @property
    ...     def template_name(self) -> str:
    ...         return "checks/sentiment.j2"
    ...
    ...     async def _build_template_inputs(self, interaction: Interaction) -> dict[str, str]:
    ...         return {"text": self.text}
    """

    @property
    @abstractmethod
    def template_name(self) -> str:
        """Return the Jinja2 template name for this check.

        Returns
        -------
        str
            Path to template relative to prompts directory.
        """
        raise NotImplementedError

    async def _build_chat_workflow(
        self, interaction: Interaction[Any, Any]
    ) -> ChatWorkflow[Any]:
        """Build ChatWorkflow using file-based template."""
        template_inputs = await self._build_template_inputs(interaction)

        return (
            self._generator.template(self.template_name)
            .with_inputs(**template_inputs)
            .with_output(self.output_type)
        )


@Check.register("inline_prompt")
class InlinePromptCheck(LLMCheck):
    """Concrete LLM check using inline template strings.

    This class can be instantiated directly with a template string,
    enabling rapid prototyping without subclassing.

    The template can access the entire `Interaction` object using Jinja2 templating,
    allowing dynamic access to inputs, outputs, and metadata fields.

    Examples
    --------
    >>> # Direct instantiation with interaction data access
    >>> check = InlinePromptCheck(
    ...     template_content="Analyze sentiment of: {{ inputs.text }}",
    ...     name="sentiment_check"
    ... )
    >>>
    >>> # Access output data
    >>> check = InlinePromptCheck(
    ...     template_content="Check if {{ outputs.text_value }} is positive",
    ...     name="output_check"
    ... )
    >>>
    >>> # Access metadata
    >>> check = InlinePromptCheck(
    ...     template_content="Analyze {{ inputs.text }} with context {{ metadata.category }}",
    ...     name="contextual_check"
    ... )
    """

    template_content: str = Field(
        ...,
        description="Inline Jinja2 template content with access to interaction data",
    )

    async def _build_chat_workflow(
        self, interaction: Interaction[Any, Any]
    ) -> ChatWorkflow[Any]:
        """Build ChatWorkflow using inline template content."""
        # Render the template with interaction data
        template_inputs = await self._build_template_inputs(interaction)
        rendered_template = template_inputs["template"]

        return self._generator.chat(rendered_template).with_output(self.output_type)

    async def _build_template_inputs(
        self, interaction: Interaction[Any, Any]
    ) -> dict[str, str]:
        """Render the template content with interaction data.

        The template can access the entire interaction object using Jinja2 templating.
        Returns a dict with the rendered template for compatibility with base class.
        """
        from jinja2 import Template

        # Render the template with the full interaction data
        rendered_template = Template(self.template_content).render(
            **interaction.model_dump()
        )

        return {"template": rendered_template}
