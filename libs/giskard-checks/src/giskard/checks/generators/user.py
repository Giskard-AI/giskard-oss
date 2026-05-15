from typing import Any, override

from giskard.agents.workflow import TemplateReference
from pydantic import Field

from ..core import Trace
from ..core.input_generator import InputGenerator
from .base import BaseLLMGenerator


@InputGenerator.register("user_simulator")
class UserSimulator[TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMGenerator[TraceType]
):
    """User simulation with predefined or custom personas.

    Accepts either a predefined persona name (e.g., "frustrated_customer") or
    a custom persona description. Extends BaseLLMGenerator with a hardcoded
    template and persona/context fields.

    Parameters
    ----------
    persona : str
        Predefined persona name or custom persona description.
    context : str | None
        Optional context to customize the persona's behavior.
    max_steps : int
        Maximum number of conversation turns (default: 3).

    Examples
    --------
    >>> simulator = UserSimulator(persona="frustrated_customer")
    >>> simulator = UserSimulator(
    ...     persona="A polite elderly user who needs step-by-step guidance",
    ...     context="Ask about using the mobile app",
    ... )
    """

    persona: str = Field(
        ..., min_length=1, description="Predefined persona name or custom description."
    )
    context: str | None = Field(
        default=None, description="Optional context to customize persona behavior."
    )

    @override
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(
            template_name="giskard.checks::generators/user_simulator.j2"
        )

    @override
    async def get_inputs(self, trace: TraceType) -> dict[str, Any]:
        return {"history": trace, "persona": self.persona, "context": self.context}
