from functools import cached_property
from typing import Any

from counterpoint.generators import Generator
from counterpoint.generators.base import BaseGenerator
from pydantic import BaseModel, Field

from giskard_checks.core.check import Check, CheckResult
from giskard_checks.core.interactions import Interaction


class LLMCheckResult(BaseModel):
    """Default result model for LLM-based checks."""

    passed: bool = Field(..., description="Whether the check passed or failed")
    reason: str | None = Field(
        default=None, description="Optional explanation for the result"
    )


class LLMCheck(Check):
    """Base class for LLM-powered validation checks.

    Provides infrastructure for checks that use LLMs to evaluate
    interactions. Subclasses must implement template_name and
    _build_template_inputs to define check-specific behavior.

    Attributes
    ----------
    generator : BaseGenerator | None
        Counterpoint generator for LLM evaluation. If None, uses global default or GPT-4o-mini.

    Examples
    --------
    >>> import counterpoint as cp
    >>> @Check.register("sentiment")
    ... class SentimentCheck(LLMCheck):
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

    @property
    def template_name(self) -> str:
        """Return the Jinja2 template name for this check.

        Returns
        -------
        str
            Path to template relative to prompts directory.
        """
        raise NotImplementedError

    async def run(
        self,
        interaction: Interaction[Any, Any],
    ) -> CheckResult:
        """Execute LLM-based validation.

        Builds template inputs, runs the LLM evaluation, and converts
        the output to a CheckResult.

        Parameters
        ----------
        interaction : Interaction
            The interaction to validate.

        Returns
        -------
        CheckResult
            Result based on LLM evaluation.
        """
        template_inputs = await self._build_template_inputs(interaction)

        generator = self._generator
        chat = await (
            generator.template(self.template_name)
            .with_inputs(**template_inputs)
            .with_output(self.output_type)
            .run()
        )

        return await self._handle_output(chat.output, template_inputs, interaction)

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
