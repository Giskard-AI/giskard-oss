from collections.abc import AsyncGenerator, Generator

import pytest
from giskard.checks import (
    BaseInteractionSpec,
    Interaction,
    InteractionSpec,
    Trace,
    UserSimulator,
)


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

    # ========== Tests for different input types ==========

    async def test_inputs_static_value(self):
        """Test inputs as static value."""
        interaction_spec = InteractionSpec(inputs="hello", outputs="hi")
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.inputs == "hello"
        assert interaction.outputs == "hi"

    async def test_inputs_fn_without_params(self):
        """Test inputs as function without parameters."""

        def get_input() -> str:
            return "hello"

        interaction_spec = InteractionSpec(inputs=get_input, outputs="hi")
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.inputs == "hello"

    async def test_inputs_fn_without_params_no_type_hint(self):
        """Test inputs as function without parameters and without type hints."""

        def get_input():
            return "hello"

        interaction_spec = InteractionSpec(inputs=get_input, outputs="hi")
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.inputs == "hello"

    async def test_inputs_fn_with_trace_param(self):
        """Test inputs as function with trace parameter."""

        def get_input(trace: Trace[str, str]) -> str:
            count = len(trace.interactions)
            return f"message_{count}"

        interaction_spec = InteractionSpec(inputs=get_input, outputs="hi")
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)
        interaction = await anext(generator)
        assert interaction.inputs == "message_0"

    async def test_inputs_fn_with_trace_param_no_type_hint(self):
        """Test that inputs function with trace parameter but no type hint works."""

        def get_input(trace):
            count = len(trace.interactions)
            return f"message_{count}"

        # Untyped parameters are now allowed and match any requirement
        interaction_spec = InteractionSpec(inputs=get_input, outputs="hi")
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)
        interaction = await anext(generator)
        assert interaction.inputs == "message_0"

    async def test_inputs_fn_with_provided_param_default(self):
        """Test inputs as function with trace and provided parameter with default."""

        def get_input(trace: Trace[str, str], provided: int = 42) -> str:
            count = len(trace.interactions)
            return f"message_{count}_{provided}"

        interaction_spec = InteractionSpec(inputs=get_input, outputs="hi")
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)
        interaction = await anext(generator)
        assert interaction.inputs == "message_0_42"

    async def test_inputs_fn_with_provided_param_default_no_type_hint(self):
        """Test that inputs function with trace and provided parameter with default but no type hints works."""

        def get_input(trace, provided=42):
            count = len(trace.interactions)
            return f"message_{count}_{provided}"

        # Untyped parameters are now allowed and match any requirement
        interaction_spec = InteractionSpec(inputs=get_input, outputs="hi")
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)
        interaction = await anext(generator)
        assert interaction.inputs == "message_0_42"

    async def test_inputs_async_fn_without_params(self):
        """Test inputs as async function without parameters."""

        async def get_input() -> str:
            return "hello"

        interaction_spec = InteractionSpec(inputs=get_input, outputs="hi")
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.inputs == "hello"

    async def test_inputs_async_fn_with_trace_param(self):
        """Test inputs as async function with trace parameter."""

        async def get_input(trace: Trace[str, str]) -> str:
            count = len(trace.interactions)
            return f"message_{count}"

        interaction_spec = InteractionSpec(inputs=get_input, outputs="hi")
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)
        interaction = await anext(generator)
        assert interaction.inputs == "message_0"

    # ========== Tests for generator inputs ==========

    async def test_inputs_sync_generator_no_params(self):
        """Test inputs as sync generator function with no parameters."""

        def inputs_generator() -> Generator[int, None, None]:
            yield 1
            yield 2
            yield 3

        interaction_spec = InteractionSpec(
            inputs=inputs_generator, outputs=lambda inputs: inputs * 2
        )
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)

        interaction = await anext(generator)
        assert interaction.inputs == 1
        assert interaction.outputs == 2

        interaction = await generator.asend(
            Trace(interactions=[*trace.interactions, interaction])
        )
        assert interaction.inputs == 2
        assert interaction.outputs == 4

        interaction = await generator.asend(
            Trace(interactions=[*trace.interactions, interaction])
        )
        assert interaction.inputs == 3
        assert interaction.outputs == 6

        with pytest.raises(StopAsyncIteration):
            await generator.asend(
                Trace(interactions=[*trace.interactions, interaction])
            )

    async def test_inputs_async_generator_no_params(self):
        """Test inputs as async generator function with no parameters."""

        async def inputs_generator() -> AsyncGenerator[int, None]:
            yield 1
            yield 2
            yield 3

        interaction_spec = InteractionSpec(
            inputs=inputs_generator, outputs=lambda inputs: inputs * 2
        )
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)

        interaction = await anext(generator)
        assert interaction.inputs == 1
        assert interaction.outputs == 2

        interaction = await generator.asend(
            Trace(interactions=[*trace.interactions, interaction])
        )
        assert interaction.inputs == 2
        assert interaction.outputs == 4

        interaction = await generator.asend(
            Trace(interactions=[*trace.interactions, interaction])
        )
        assert interaction.inputs == 3
        assert interaction.outputs == 6

        with pytest.raises(StopAsyncIteration):
            await generator.asend(
                Trace(interactions=[*trace.interactions, interaction])
            )

    async def test_inputs_sync_generator_with_trace_param(self):
        """Test inputs as sync generator function with trace parameter."""

        def inputs_generator(
            trace: Trace[int, int],
        ) -> Generator[int, Trace[int, int], None]:
            count = len(trace.interactions)
            trace = yield count + 1
            trace = yield count + 2
            trace = yield count + 3

        interaction_spec = InteractionSpec(
            inputs=inputs_generator, outputs=lambda inputs: inputs * 2
        )
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)

        interaction = await anext(generator)
        assert interaction.inputs == 1
        assert interaction.outputs == 2

        updated_trace = Trace(interactions=[*trace.interactions, interaction])
        interaction = await generator.asend(updated_trace)
        assert interaction.inputs == 2
        assert interaction.outputs == 4

        updated_trace = Trace(interactions=[*updated_trace.interactions, interaction])
        interaction = await generator.asend(updated_trace)
        assert interaction.inputs == 3
        assert interaction.outputs == 6

        with pytest.raises(StopAsyncIteration):
            await generator.asend(
                Trace(interactions=[*updated_trace.interactions, interaction])
            )

    async def test_inputs_async_generator_with_trace_param(self):
        """Test inputs as async generator function with trace parameter."""

        async def inputs_generator(
            trace: Trace[int, int],
        ) -> AsyncGenerator[int, Trace[int, int]]:
            count = len(trace.interactions)
            trace = yield count + 1
            trace = yield count + 2
            trace = yield count + 3

        interaction_spec = InteractionSpec(
            inputs=inputs_generator, outputs=lambda inputs: inputs * 2
        )
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)

        interaction = await anext(generator)
        assert interaction.inputs == 1
        assert interaction.outputs == 2

        updated_trace = Trace(interactions=[*trace.interactions, interaction])
        interaction = await generator.asend(updated_trace)
        assert interaction.inputs == 2
        assert interaction.outputs == 4

        updated_trace = Trace(interactions=[*updated_trace.interactions, interaction])
        interaction = await generator.asend(updated_trace)
        assert interaction.inputs == 3
        assert interaction.outputs == 6

        with pytest.raises(StopAsyncIteration):
            await generator.asend(
                Trace(interactions=[*updated_trace.interactions, interaction])
            )

    async def test_inputs_sync_generator_no_params_no_type_hint(self):
        """Test inputs as sync generator function with no parameters and no type hints."""

        def inputs_generator():
            yield 1
            yield 2

        interaction_spec = InteractionSpec(
            inputs=inputs_generator, outputs=lambda inputs: inputs * 2
        )
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)

        interaction = await anext(generator)
        assert interaction.inputs == 1
        assert interaction.outputs == 2

        interaction = await generator.asend(
            Trace(interactions=[*trace.interactions, interaction])
        )
        assert interaction.inputs == 2
        assert interaction.outputs == 4

        with pytest.raises(StopAsyncIteration):
            await generator.asend(
                Trace(interactions=[*trace.interactions, interaction])
            )

    async def test_inputs_async_generator_no_params_no_type_hint(self):
        """Test inputs as async generator function with no parameters and no type hints."""

        async def inputs_generator():
            yield 1
            yield 2

        interaction_spec = InteractionSpec(
            inputs=inputs_generator, outputs=lambda inputs: inputs * 2
        )
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)

        interaction = await anext(generator)
        assert interaction.inputs == 1
        assert interaction.outputs == 2

        interaction = await generator.asend(
            Trace(interactions=[*trace.interactions, interaction])
        )
        assert interaction.inputs == 2
        assert interaction.outputs == 4

        with pytest.raises(StopAsyncIteration):
            await generator.asend(
                Trace(interactions=[*trace.interactions, interaction])
            )

    async def test_inputs_sync_generator_with_trace_no_type_hint(self):
        """Test inputs as sync generator function with trace parameter but no type hints."""

        def inputs_generator(trace):
            count = len(trace.interactions)
            trace = yield count + 1
            trace = yield count + 2

        interaction_spec = InteractionSpec(
            inputs=inputs_generator, outputs=lambda inputs: inputs * 2
        )
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)

        interaction = await anext(generator)
        assert interaction.inputs == 1
        assert interaction.outputs == 2

        updated_trace = Trace(interactions=[*trace.interactions, interaction])
        interaction = await generator.asend(updated_trace)
        assert interaction.inputs == 2
        assert interaction.outputs == 4

        with pytest.raises(StopAsyncIteration):
            await generator.asend(
                Trace(interactions=[*updated_trace.interactions, interaction])
            )

    async def test_inputs_async_generator_with_trace_no_type_hint(self):
        """Test inputs as async generator function with trace parameter but no type hints."""

        async def inputs_generator(trace):
            count = len(trace.interactions)
            trace = yield count + 1
            trace = yield count + 2

        interaction_spec = InteractionSpec(
            inputs=inputs_generator, outputs=lambda inputs: inputs * 2
        )
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)

        interaction = await anext(generator)
        assert interaction.inputs == 1
        assert interaction.outputs == 2

        updated_trace = Trace(interactions=[*trace.interactions, interaction])
        interaction = await generator.asend(updated_trace)
        assert interaction.inputs == 2
        assert interaction.outputs == 4

        with pytest.raises(StopAsyncIteration):
            await generator.asend(
                Trace(interactions=[*updated_trace.interactions, interaction])
            )

    # ========== Tests for different output types ==========

    async def test_outputs_static_value(self):
        """Test outputs as static value."""
        interaction_spec = InteractionSpec(inputs="hello", outputs="hi")
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.outputs == "hi"

    async def test_outputs_fn_without_params(self):
        """Test outputs as function without parameters."""

        def get_output() -> str:
            return "hi"

        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.outputs == "hi"

    async def test_outputs_fn_without_params_no_type_hint(self):
        """Test outputs as function without parameters and without type hints."""

        def get_output():
            return "hi"

        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.outputs == "hi"

    async def test_outputs_fn_with_inputs_param(self):
        """Test outputs as function with inputs parameter."""

        def get_output(inputs: str) -> str:
            return f"echo: {inputs}"

        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.outputs == "echo: hello"

    async def test_outputs_fn_with_inputs_param_no_type_hint(self):
        """Test outputs as function with inputs parameter but no type hint."""

        def get_output(inputs):
            return f"echo: {inputs}"

        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.outputs == "echo: hello"

    async def test_outputs_fn_with_inputs_and_trace_params(self):
        """Test outputs as function with inputs and trace parameters."""

        def get_output(inputs: str, trace: Trace[str, str]) -> str:
            count = len(trace.interactions)
            return f"echo_{count}: {inputs}"

        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)
        interaction = await anext(generator)
        assert interaction.outputs == "echo_0: hello"

    async def test_outputs_fn_with_inputs_and_trace_params_no_type_hint(self):
        """Test outputs as function with inputs and trace parameters but no type hints."""

        def get_output(inputs, trace):
            count = len(trace.interactions)
            return f"echo_{count}: {inputs}"

        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)
        interaction = await anext(generator)
        assert interaction.outputs == "echo_0: hello"

    async def test_outputs_fn_with_provided_param_default(self):
        """Test outputs as function with inputs, trace, and provided parameter with default."""

        def get_output(inputs: str, trace: Trace[str, str], multiplier: int = 2) -> str:
            count = len(trace.interactions)
            return f"echo_{count * multiplier}: {inputs}"

        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)
        interaction = await anext(generator)
        assert interaction.outputs == "echo_0: hello"

    async def test_outputs_fn_with_provided_param_default_no_type_hint(self):
        """Test outputs as function with provided parameter with default, no type hints."""

        def get_output(inputs, trace, multiplier=2):
            count = len(trace.interactions)
            return f"echo_{count * multiplier}: {inputs}"

        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)
        interaction = await anext(generator)
        assert interaction.outputs == "echo_0: hello"

    async def test_outputs_async_fn_with_inputs_param(self):
        """Test outputs as async function with inputs parameter."""

        async def get_output(inputs: str) -> str:
            return f"echo: {inputs}"

        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.outputs == "echo: hello"

    async def test_outputs_async_fn_with_inputs_and_trace_params(self):
        """Test outputs as async function with inputs and trace parameters."""

        async def get_output(inputs: str, trace: Trace[str, str]) -> str:
            count = len(trace.interactions)
            return f"echo_{count}: {inputs}"

        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)
        interaction = await anext(generator)
        assert interaction.outputs == "echo_0: hello"

    # ========== Tests for validation errors ==========

    async def test_inputs_fn_with_unmapped_required_param_raises_error(self):
        """Test that inputs function with unmapped required parameter raises validation error."""

        def get_input(trace: Trace[str, str], unmapped: str) -> str:
            return f"message_{unmapped}"

        # TypeError is raised directly (not wrapped because code only catches ValueError)
        with pytest.raises(TypeError, match="Parameter 'unmapped'.*no matching"):
            InteractionSpec(inputs=get_input, outputs="hi")

    async def test_outputs_fn_with_unmapped_required_param_passes_validation_but_fails_runtime(
        self,
    ):
        """Test that outputs function with unmapped required parameter passes validation but fails at runtime.

        Note: Since INJECTABLE_INPUT has class_info=Any, it matches any parameter type,
        so validation passes. However, at runtime it fails because the unmapped parameter
        cannot be resolved from the provided arguments.
        """

        def get_output(inputs: str, trace: Trace[str, str], unmapped: str) -> str:
            # unmapped will match INJECTABLE_INPUT (Any) during validation
            return f"echo_{unmapped}: {inputs}"

        # Validation passes because INJECTABLE_INPUT (Any) matches unmapped: str
        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        generator = interaction_spec.generate(Trace(interactions=[]))
        # Fails at runtime with IndexError because unmapped parameter (position 2)
        # cannot be resolved from args (only inputs and trace are provided)
        with pytest.raises(IndexError):
            await anext(generator)

    async def test_inputs_fn_with_var_positional_works(self):
        """Test that inputs function with *args works (var args are skipped)."""

        def get_input(*args) -> str:
            return "hello"

        # *args are skipped in parameter injection, so this should work
        interaction_spec = InteractionSpec(inputs=get_input, outputs="hi")
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.inputs == "hello"

    async def test_outputs_fn_with_var_positional_works(self):
        """Test that outputs function with *args works (var args are skipped)."""

        def get_output(*args) -> str:
            return "hi"

        # *args are skipped in parameter injection, so this should work
        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.outputs == "hi"

    async def test_inputs_fn_with_var_keyword_works(self):
        """Test that inputs function with **kwargs works (var kwargs are skipped)."""

        def get_input(**kwargs) -> str:
            return "hello"

        # **kwargs are skipped in parameter injection, so this should work
        interaction_spec = InteractionSpec(inputs=get_input, outputs="hi")
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.inputs == "hello"

    async def test_outputs_fn_with_var_keyword_works(self):
        """Test that outputs function with **kwargs works (var kwargs are skipped)."""

        def get_output(**kwargs) -> str:
            return "hi"

        # **kwargs are skipped in parameter injection, so this should work
        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        generator = interaction_spec.generate(Trace(interactions=[]))
        interaction = await anext(generator)
        assert interaction.outputs == "hi"

    # ========== Combined tests ==========

    async def test_combined_static_inputs_fn_outputs_with_trace(self):
        """Test static inputs with function outputs that use trace."""

        def get_output(inputs: str, trace: Trace[str, str]) -> str:
            count = len(trace.interactions)
            return f"[{count}] {inputs}"

        interaction_spec = InteractionSpec(inputs="hello", outputs=get_output)
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)
        interaction = await anext(generator)
        assert interaction.inputs == "hello"
        assert interaction.outputs == "[0] hello"

    async def test_combined_fn_inputs_with_trace_static_outputs(self):
        """Test function inputs with trace and static outputs."""

        def get_input(trace: Trace[str, str]) -> str:
            count = len(trace.interactions)
            return f"message_{count}"

        interaction_spec = InteractionSpec(inputs=get_input, outputs="hi")
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)
        interaction = await anext(generator)
        assert interaction.inputs == "message_0"
        assert interaction.outputs == "hi"

    async def test_combined_fn_inputs_with_provided_fn_outputs_with_provided(self):
        """Test function inputs with provided param and function outputs with provided param."""

        def get_input(trace: Trace[str, str], offset: int = 10) -> str:
            count = len(trace.interactions)
            return f"message_{count + offset}"

        def get_output(inputs: str, trace: Trace[str, str], multiplier: int = 3) -> str:
            count = len(trace.interactions)
            return f"[{count * multiplier}] {inputs}"

        interaction_spec = InteractionSpec(inputs=get_input, outputs=get_output)
        trace = Trace(interactions=[])
        generator = interaction_spec.generate(trace)
        interaction = await anext(generator)
        assert interaction.inputs == "message_10"
        assert interaction.outputs == "[0] message_10"

    # ========== Tests for serialization ==========

    def test_interaction_spec_serialization_with_user_simulator_inputs(self):
        """Test that InteractionSpec with UserSimulator inputs and static outputs can be serialized and deserialized."""
        user_simulator = UserSimulator(
            instructions="Ask about the weather", max_steps=2
        )
        interaction_spec = InteractionSpec(
            inputs=user_simulator, outputs="This is a static response"
        )

        # Serialize to JSON
        json_str = interaction_spec.model_dump_json()

        # Deserialize from JSON
        restored_spec = BaseInteractionSpec.model_validate_json(json_str)

        # Verify it's an InteractionSpec
        assert isinstance(restored_spec, InteractionSpec)

        # Verify the UserSimulator was restored correctly
        assert isinstance(restored_spec.inputs, UserSimulator)
        assert restored_spec.inputs.instructions == "Ask about the weather"
        assert restored_spec.inputs.max_steps == 2

        # Verify the static output was preserved
        assert restored_spec.outputs == "This is a static response"
