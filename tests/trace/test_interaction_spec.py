from collections.abc import Generator

import pytest
from giskard.checks.core import Interaction, Trace
from giskard.checks.interaction import InteractionSpec


class TestInteractionSpec:
    async def test_interaction_spec_with_static_inputs_and_outputs(self):
        interaction_spec = InteractionSpec(inputs=1, outputs=2)

        generator = interaction_spec.generate(Trace(interactions=[]))

        interaction = await anext(generator)
        assert interaction == Interaction(inputs=1, outputs=2, metadata={})
        with pytest.raises(StopAsyncIteration):
            await generator.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=2, metadata={})])
            )

    async def test_interaction_spec_with_dynamic_inputs_and_outputs(self):
        interaction_spec = InteractionSpec(
            inputs=lambda: 1, outputs=lambda inputs: inputs + 1
        )

        generator = interaction_spec.generate(Trace(interactions=[]))

        interaction = await anext(generator)
        assert interaction == Interaction(inputs=1, outputs=2, metadata={})
        with pytest.raises(StopAsyncIteration):
            await generator.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=2, metadata={})])
            )

    async def test_interaction_spec_with_inputs_generator(self):
        def inputs_generator(
            trace: Trace[int, int],
        ) -> Generator[int, Trace[int, int], None]:
            trace = yield 1
            trace = yield trace.interactions[-1].outputs + 1
            trace = yield trace.interactions[-1].outputs + 1

        interaction_spec = InteractionSpec(
            inputs=inputs_generator, outputs=lambda inputs: inputs + 1
        )

        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)

        interaction = await anext(generator)
        assert interaction == Interaction(inputs=1, outputs=2, metadata={})
        interaction = await generator.asend(
            Trace(interactions=[*trace.interactions, interaction])
        )
        assert interaction == Interaction(inputs=3, outputs=4, metadata={})
        interaction = await generator.asend(
            Trace(interactions=[*trace.interactions, interaction])
        )
        assert interaction == Interaction(inputs=5, outputs=6, metadata={})
        with pytest.raises(StopAsyncIteration):
            await generator.asend(
                Trace(interactions=[*trace.interactions, interaction])
            )

    async def test_interaction_spec_with_inputs_generator_custom_trace(self):
        class CustomTrace(Trace[int, int], frozen=True):
            def outputs(self) -> int:
                return self.interactions[-1].outputs if self.interactions else 0

        def inputs_generator(trace: CustomTrace) -> Generator[int, CustomTrace, None]:
            trace = yield trace.outputs() + 1
            trace = yield trace.outputs() + 1
            trace = yield trace.outputs() + 1

        interaction_spec = InteractionSpec(
            inputs=inputs_generator, outputs=lambda inputs: inputs + 1
        )

        trace = CustomTrace()
        generator = interaction_spec.generate(trace)

        interaction = await anext(generator)
        assert interaction == Interaction(inputs=1, outputs=2, metadata={})
        interaction = await generator.asend(await trace.with_interaction(interaction))
        assert interaction == Interaction(inputs=3, outputs=4, metadata={})
        interaction = await generator.asend(await trace.with_interaction(interaction))
        assert interaction == Interaction(inputs=5, outputs=6, metadata={})
        with pytest.raises(StopAsyncIteration):
            await generator.asend(await trace.with_interaction(interaction))
