from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

from counterpoint.generators import Generator
from counterpoint.generators.base import BaseGenerator
from counterpoint.workflow import ChatWorkflow
from pydantic import BaseModel, Field

from giskard_checks.core.check import Check, CheckResult
from giskard_checks.core.interaction_result import InteractionResult


class LLMCheckResult(BaseModel):
    """Default result model for LLM-based checks."""

    passed: bool = Field(..., description="Whether the check passed or failed")
    reason: str | None = Field(
        default=None, description="Optional explanation for the result"
    )


class LLMCheck(Check, ABC):
    """Abstract base class for LLM-powered validation checks.

    Provides infrastructure for checks that use LLMs to evaluate
    interaction results. Subclasses must implement _build_chat_workflow and
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
    ...     async def _build_template_inputs(self, interaction: InteractionResult) -> dict[str, str]:
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
        self, interaction: InteractionResult[Any, Any]
    ) -> ChatWorkflow[Any]:
        """Build the ChatWorkflow for this check.

        Subclasses must implement this to create the appropriate
        ChatWorkflow (file-based or inline template).

        Parameters
        ----------
        interaction : InteractionResult
            The interaction result to validate.

        Returns
        -------
        ChatWorkflow
            Configured workflow ready to run
        """
        raise NotImplementedError

    async def run(
        self,
        interaction: InteractionResult[Any, Any],
    ) -> CheckResult:
        """Execute LLM-based validation using the configured ChatWorkflow.

        Parameters
        ----------
        interaction : InteractionResult
            The interaction result to validate.

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
        self, interaction: InteractionResult[Any, Any]
    ) -> dict[str, str]:
        """Build template variables from the interaction result.

        Subclasses must implement this to extract values for their
        specific template.

        Parameters
        ----------
        interaction : InteractionResult
            The interaction result containing inputs, outputs, and metadata.

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
        interaction: InteractionResult[Any, Any],
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
        interaction : InteractionResult
            The original interaction result.

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
        self, interaction: InteractionResult[Any, Any]
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
    """Concrete LLM check using inline Jinja2 template strings.

    This class can be instantiated directly with a Jinja2 template string,
    enabling rapid prototyping without subclassing. The template has direct
    access to the interaction's inputs, outputs, and metadata.

    Template Variables
    ------------------
    inputs : Any
        The input payload for the system under test.
    outputs : Any | None
        The output produced by the system (may be None).
    metadata : dict[str, Any] | None
        Optional free-form metadata associated with the interaction.

    Parameters
    ----------
    template_content : str
        Jinja2 template string that will be rendered with interaction data.
        The template is rendered before being sent to the LLM for evaluation.
    name : str
        Name identifier for this check instance.
    generator : BaseGenerator | None, optional
        Counterpoint generator for LLM evaluation. If None, uses global default.

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
    >>>
    >>> # Access metadata for context
    >>> check = InlinePromptCheck(
    ...     template_content="Analyze '{{ inputs }}' for category: {{ metadata.category }}",
    ...     name="contextual_check"
    ... )
    >>>
    >>> # Use Jinja2 filters and control structures
    >>> check = InlinePromptCheck(
    ...     template_content='''
    ...     {% if metadata.language == "fr" %}
    ...     Vérifier si "{{ outputs }}" est poli.
    ...     {% else %}
    ...     Check if "{{ outputs }}" is polite.
    ...     {% endif %}
    ...     ''',
    ...     name="politeness_check"
    ... )

    Notes
    -----
    - The template is rendered using Jinja2's Template.render() with the full
      interaction data (inputs, outputs, metadata).
    - If a variable is not present in the interaction, Jinja2 will render it as
      an empty string by default (undefined variables don't raise errors).
    - All standard Jinja2 features are supported: filters, tests, control
      structures (if/for), whitespace control, etc.
    """

    template_content: str = Field(
        ...,
        description="Inline Jinja2 template content with access to interaction data",
    )

    async def _build_chat_workflow(
        self, interaction: InteractionResult[Any, Any]
    ) -> ChatWorkflow[Any]:
        """Build ChatWorkflow using inline template content."""
        # Render the template with interaction data
        template_inputs = await self._build_template_inputs(interaction)
        rendered_template = template_inputs["template"]

        return self._generator.chat(rendered_template).with_output(self.output_type)

    async def _build_template_inputs(
        self, interaction: InteractionResult[Any, Any]
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
