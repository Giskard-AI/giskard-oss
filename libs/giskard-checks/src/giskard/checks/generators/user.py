from collections.abc import AsyncGenerator

from pydantic import BaseModel, Field

from ..core import Trace
from ..core.input_generator import InputGenerator
from ..core.mixin import WithGeneratorMixin


class UserSimulatorOutput(BaseModel):
    goal_reached: bool = Field(
        ...,
        description="Whether the goal has been reached. Meaning that the persona's goal has been achieved and no more messages are needed.",
    )
    message: str | None = Field(
        default=None,
        description="The message that the user would send. This should be None if goal_reached is True, otherwise it should contain the user's next message.",
    )


@InputGenerator.register("user_simulator")
class UserSimulator[TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    InputGenerator[str, TraceType], WithGeneratorMixin
):
    """User simulation with predefined or custom personas.

    Accepts either a predefined persona name (e.g., "frustrated_customer") or a custom
    persona description.

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
    Predefined persona:

    >>> simulator = UserSimulator(persona="frustrated_customer")

    Custom persona with optional context:

    >>> simulator = UserSimulator(
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

    async def __call__(self, trace: TraceType) -> AsyncGenerator[str, TraceType]:
        user_generator_workflow_ = (
            self.generator.template("giskard.checks::generators/user_simulator.j2")
            .with_inputs(persona=self.persona, context=self.context)
            .with_output(UserSimulatorOutput)
        )

        step = 0
        while step < self.max_steps:
            chat = await user_generator_workflow_.with_inputs(history=trace).run()
            output = chat.output

            if output.goal_reached or not output.message:
                return

            trace = yield output.message
            step += 1
