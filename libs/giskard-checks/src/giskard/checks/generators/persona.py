from collections.abc import AsyncGenerator
from typing import override

from pydantic import BaseModel, Field, PrivateAttr

from ..core.input_generator import InputGenerator
from ..core.mixin import WithGeneratorMixin
from ..core.trace import Trace


class PersonaSimulatorOutput(BaseModel):
    """Output from PersonaSimulator including client description for consistency."""

    client_description: str | None = Field(
        default=None,
        description="Detailed description of the specific client being simulated. Generated on first turn, None on subsequent turns.",
    )
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
    """User simulation with custom personas.

    Parameters
    ----------
    persona : str
        Persona description (e.g., "A polite elderly user who needs step-by-step guidance")
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

    persona: str = Field(..., description="Persona description", min_length=1)
    context: str | None = Field(
        default=None, description="Optional context to customize persona behavior"
    )
    max_steps: int = Field(default=3, ge=0)

    # Client description generated on first turn and reused
    _client_description: str | None = PrivateAttr(default=None)

    @override
    async def __call__(self, trace: TraceType) -> AsyncGenerator[str, TraceType]:
        persona_generator_workflow_ = self.generator.template(
            "giskard.checks::generators/persona_simulator.j2"
        ).with_output(PersonaSimulatorOutput)

        step = 0
        while step < self.max_steps:
            # First turn: no client description yet
            # Subsequent turns: pass existing client description
            chat = await persona_generator_workflow_.with_inputs(
                persona=self.persona,
                context=self.context,
                client_description=self._client_description,
                history=trace,
            ).run()

            output = chat.output

            # Store client description from first turn
            if output.client_description and not self._client_description:
                self._client_description = output.client_description

            if output.goal_reached or not output.message:
                return

            trace = yield output.message
            step += 1
