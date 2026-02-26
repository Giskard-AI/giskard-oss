from collections.abc import AsyncGenerator

from pydantic import BaseModel, Field

from ..core import Trace
from ..core.input_generator import InputGenerator
from ..core.mixin import WithGeneratorMixin


class UserSimulatorOutput(BaseModel):
    goal_reached: bool = Field(
        ...,
        description="Whether the goal has been reached. Meanining that the instructions have been followed and no more messages are needed.",
    )
    message: str | None = Field(
        default=None,
        description="The message that the user would send. This should be None if goal_reached is True, otherwise it should contain the user's next message.",
    )


@InputGenerator.register("user_simulator")
class UserSimulator[TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    InputGenerator[str, TraceType], WithGeneratorMixin
):
    instructions: str
    max_steps: int = Field(default=3)

    async def __call__(self, trace: TraceType) -> AsyncGenerator[str, TraceType]:
        user_generator_workflow_ = (
            self.generator.template("giskard.checks::generators/user_simulator.j2")
            .with_inputs(instructions=self.instructions)
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
