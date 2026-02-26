from collections.abc import AsyncGenerator
from typing import override
from unittest.mock import MagicMock, patch

from ..core import Trace
from ..core.interaction import Interaction, InteractionSpec


@InteractionSpec.register("with_spy")
class WithSpy[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    InteractionSpec[InputType, OutputType, TraceType]
):
    interaction_generator: InteractionSpec[InputType, OutputType, TraceType]
    target: str

    @override
    async def generate(
        self, trace: TraceType
    ) -> AsyncGenerator[Interaction[InputType, OutputType], TraceType]:
        spy = MagicMock()
        with patch(self.target, spy):
            generator = self.interaction_generator.generate(trace)

            try:
                trace = yield self._patch_interaction(await anext(generator), spy)
                while True:
                    trace = yield self._patch_interaction(
                        await generator.asend(trace), spy
                    )
            except StopAsyncIteration:
                return
            finally:
                await generator.aclose()

    def _patch_interaction(
        self, interaction: Interaction[InputType, OutputType], spy: MagicMock
    ) -> Interaction[InputType, OutputType]:
        interaction = Interaction(
            inputs=interaction.inputs,
            outputs=interaction.outputs,
            metadata=interaction.metadata
            | {
                self.target: {
                    "call_args_list": spy.call_args_list,
                    "call_count": spy.call_count,
                    "call_args": spy.call_args,
                    "mock_calls": spy.mock_calls,
                }
            },
        )
        spy.reset_mock()
        return interaction
