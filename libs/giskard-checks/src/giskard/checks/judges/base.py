from typing import Any, override

from giskard.agents import (
    MessageTemplate,
    TemplateReference,
)
from giskard.llm.types import ChatMessage
from pydantic import BaseModel

from .._judge_result import LLMCheckResult as LLMCheckResult
from ..core import Trace
from ..core.check import Check
from ..core.mixin import WithJudgeMixin
from ..core.result import CheckResult


def format_prompt_text(value: Any) -> str:
    """Render a resolved trace/input value as plain text for an LLM prompt.

    Judge inputs such as ``context`` accept ``str | list[str]``, and
    JSONPath resolution can also return a list for ``answer`` (multi-match
    paths, or a single match whose value is already a list). A bare
    ``str(value)`` on a list renders Python's ``repr`` -- brackets, comma
    separators, and quoted (sometimes escaped) items -- which leaks
    implementation syntax into the prompt. Join list values into a single
    newline-separated block instead. Any other value (a plain string,
    ``NoMatch``, or arbitrary trace payload) is stringified as before.
    """
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


class BaseLLMCheck[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType], WithJudgeMixin
):
    """Abstract base class for LLM-based checks.

    Provides a framework for creating checks that use a :class:`BaseJudge`
    backend (LLM chat or System One Model) to evaluate interactions.
    Subclasses must implement the `get_prompt` method to define how the
    evaluation prompt is built.

    Attributes
    ----------
    judge : BaseJudge or None
        Judge backend. Defaults to the global default judge when unset.
        Legacy ``generator=`` kwargs are migrated to ``judge`` automatically.
    """

    @property
    def output_type(self) -> type[BaseModel] | None:
        return LLMCheckResult

    def get_prompt(self) -> str | ChatMessage | MessageTemplate | TemplateReference:
        """Get the prompt for the LLM evaluation.

        Returns
        -------
        str | ChatMessage | MessageTemplate | TemplateReference
            The prompt to send to the LLM. Can be:
            - A string (converted to MessageTemplate)
            - A ChatMessage object
            - A MessageTemplate object
            - A TemplateReference for file-based templates
        """
        raise NotImplementedError

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
        inputs = await self.get_inputs(trace)
        output = await self._judge.judge(
            self.get_prompt(), inputs, output_type=self.output_type
        )
        return await self._handle_output(output, inputs, trace)

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
                    message=output_value.reason,
                    details={
                        "reason": output_value.reason,
                        "inputs": template_inputs,
                    },
                )
            else:
                return CheckResult.failure(
                    message=output_value.reason,
                    details={
                        "reason": output_value.reason,
                        "inputs": template_inputs,
                    },
                )

        raise NotImplementedError(
            f"Custom output type {type(output_value)} requires overriding _handle_output"
        )
