from collections.abc import AsyncGenerator
from typing import override

from pydantic import BaseModel, Field

from ..core.input_generator import InputGenerator
from ..core.mixin import WithGeneratorMixin
from ..core.trace import Trace


class PersonaSimulatorOutput(BaseModel):
    """Output from PersonaSimulator."""

    goal_reached: bool = Field(
        ...,
        description="Whether the goal has been reached.",
    )
    message: str | None = Field(
        default=None,
        description="The message from this client. None if goal_reached is True.",
    )


@InputGenerator.register("persona_simulator")
class PersonaSimulator[TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    InputGenerator[str, TraceType], WithGeneratorMixin
):
    """User simulation with predefined or custom personas.

    Accepts either a predefined persona name (e.g., "frustrated_customer") or a custom
    persona description. No client description is generated; the persona is used directly.

    Parameters
    ----------
    persona : str
        Predefined persona name or custom persona description
    context : str | None
        Optional context to customize the persona's behavior
    max_steps : int
        Maximum number of conversation turns (default: 3)

    Examples
    --------
    >>> simulator = PersonaSimulator(
    ...     persona="A polite elderly user who needs step-by-step guidance",
    ...     context="Ask about using the mobile app"
    ... )
    """

    persona: str = Field(
        ..., description="Predefined persona name or custom description", min_length=1
    )
    context: str | None = Field(
        default=None, description="Optional context to customize persona behavior"
    )
    max_steps: int = Field(default=3, ge=0)

    @override
    async def __call__(self, trace: TraceType) -> AsyncGenerator[str, TraceType]:
        persona_generator_workflow_ = self.generator.template(
            "giskard.checks::generators/persona_simulator.j2"
        ).with_output(PersonaSimulatorOutput)

        step = 0
        while step < self.max_steps:
            chat = await persona_generator_workflow_.with_inputs(
                persona=self.persona,
                context=self.context,
                history=trace,
            ).run()

            output = chat.output

            if output.goal_reached or not output.message:
                return

            trace = yield output.message
            step += 1
