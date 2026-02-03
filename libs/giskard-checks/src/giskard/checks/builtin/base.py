from typing import Any, override

from giskard.agents.chat import Message
from giskard.agents.templates import MessageTemplate
from giskard.agents.workflow import ChatWorkflow, TemplateReference
from pydantic import BaseModel, Field

from ..core.check import Check
from ..core.mixin import WithGeneratorMixin
from ..core.result import CheckResult
from ..core.trace import Trace


class LLMCheckResult(BaseModel):
    """Default result model for LLM-based checks."""

    reason: str | None = Field(
        default=None, description="Optional explanation for the result"
    )
    passed: bool = Field(..., description="Whether the check passed or failed")


class BaseLLMCheck[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType], WithGeneratorMixin
):
    """Abstract base class for LLM-based checks.

    Provides a framework for creating checks that use Large Language Models
    to evaluate interactions. Subclasses must implement the `get_prompt` method
    to define how the LLM should be prompted.

    Attributes
    ----------
    generator : BaseGenerator
        Generator for LLM evaluation. Defaults to the global
        default generator if not specified.
    """

    @property
    def output_type(self) -> type[BaseModel] | None:
        return LLMCheckResult

    def get_prompt(self) -> str | Message | MessageTemplate | TemplateReference:
        """Get the prompt for the LLM evaluation.

        Returns
        -------
        str | Message | MessageTemplate | TemplateReference
            The prompt to send to the LLM. Can be:
            - A string (converted to MessageTemplate)
            - A Message object
            - A MessageTemplate object
            - A TemplateReference for file-based templates
        """
        raise NotImplementedError

    async def _build_workflow(self, trace: TraceType) -> ChatWorkflow[Any]:
        """Build the workflow for LLM evaluation.

        Parameters
        ----------
        trace : Trace
            The trace to evaluate.

        Returns
        -------
        ChatWorkflow[Any]
            Configured workflow ready for execution.
        """
        _ = trace  # Not used in base implementation
        prompt = self.get_prompt()

        if isinstance(prompt, str):
            prompt = MessageTemplate(role="user", content_template=prompt)

        return ChatWorkflow(generator=self.generator, messages=[prompt])

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the LLM-based check.

        Parameters
        ----------
        trace : Trace
            The trace containing interaction history. Access the current
            interaction via `trace.last` (preferred in prompt templates) or
            `trace.interactions[-1]` if available.

        Returns
        -------
        CheckResult
            The result of the check evaluation.
        """
        workflow = await self._build_workflow(trace)

        inputs = await self.get_inputs(trace)
        workflow = workflow.with_inputs(**inputs)

        if self.output_type is not None:
            workflow = workflow.with_output(self.output_type)

        chat = await workflow.run()

        return await self._handle_output(chat.output, inputs, trace)

    async def get_inputs(self, trace: TraceType) -> dict[str, Any]:
        """Get template inputs for the LLM prompt.

        Parameters
        ----------
        trace : Trace
            The trace containing interaction history.

        Returns
        -------
        dict[str, Any]
            Template variables available in the prompt. Default implementation
            provides the trace object under the 'trace' key, allowing templates
            to access properties like `trace.interactions` and `trace.last`.
        """
        return {"trace": trace}

    async def _handle_output(
        self,
        output_value: BaseModel,
        template_inputs: dict[str, Any],
        trace: TraceType,
    ) -> CheckResult:
        """Convert LLM output to CheckResult.

        Default implementation handles LLMCheckResult. Override for
        custom output types.

        Parameters
        ----------
        output_value : BaseModel
            The structured output from the LLM.
        template_inputs : dict[str, Any]
            The template inputs used for the evaluation.
        trace : Trace
            The original trace.

        Returns
        -------
        CheckResult
            Success or failure based on LLM output.
        """
        _ = trace  # Not used in base implementation
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
