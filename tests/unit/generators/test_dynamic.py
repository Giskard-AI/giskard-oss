"""Tests for DynamicInteraction generator."""

from giskard.checks.core.context import Context
from giskard.checks.core.interaction_result import InteractionResult
from giskard.checks.generators import DynamicInteraction


class TestDynamicInteraction:
    """Test cases for DynamicInteraction generator."""

    async def test_sync_callable_without_context(self):
        """Test sync callable that doesn't take context parameter."""

        def simple_generator():
            return InteractionResult(inputs="hello", outputs="world")

        interaction = DynamicInteraction(fn=simple_generator)
        result = await interaction.generate(Context())

        assert result.inputs == "hello"
        assert result.outputs == "world"
        assert result.metadata is None

    async def test_sync_callable_with_context(self):
        """Test sync callable that takes context parameter."""

        def context_generator(context: Context):
            return InteractionResult(
                inputs=len(context.previous_interactions),
                outputs="processed",
                metadata={"context_length": len(context.previous_interactions)},
            )

        context = Context(
            previous_interactions=[
                InteractionResult(inputs="test1", outputs="result1"),
                InteractionResult(inputs="test2", outputs="result2"),
            ]
        )

        interaction = DynamicInteraction(fn=context_generator)
        result = await interaction.generate(context)

        assert result.inputs == 2
        assert result.outputs == "processed"
        assert result.metadata == {"context_length": 2}

    async def test_async_callable_without_context(self):
        """Test async callable that doesn't take context parameter."""

        async def async_generator():
            return InteractionResult(inputs="async_hello", outputs="async_world")

        interaction = DynamicInteraction(fn=async_generator)
        result = await interaction.generate(Context())

        assert result.inputs == "async_hello"
        assert result.outputs == "async_world"

    async def test_async_callable_with_context(self):
        """Test async callable that takes context parameter."""

        async def async_context_generator(context: Context):
            return InteractionResult(
                inputs=[i.inputs for i in context.previous_interactions],
                outputs="async_processed",
            )

        context = Context(
            previous_interactions=[
                InteractionResult(inputs="msg1", outputs="resp1"),
                InteractionResult(inputs="msg2", outputs="resp2"),
            ]
        )

        interaction = DynamicInteraction(fn=async_context_generator)
        result = await interaction.generate(context)

        assert result.inputs == ["msg1", "msg2"]
        assert result.outputs == "async_processed"

    async def test_metadata_passing(self):
        """Test that metadata is properly passed through."""

        def metadata_generator():
            return InteractionResult(
                inputs="test",
                outputs="result",
                metadata={"custom": "data", "number": 42},
            )

        interaction = DynamicInteraction(fn=metadata_generator)
        result = await interaction.generate(Context())

        assert result.metadata == {"custom": "data", "number": 42}

    async def test_empty_context(self):
        """Test with empty context."""

        def empty_context_generator(context: Context):
            return InteractionResult(
                inputs=len(context.previous_interactions),
                outputs="empty"
                if len(context.previous_interactions) == 0
                else "not_empty",
            )

        interaction = DynamicInteraction(fn=empty_context_generator)
        result = await interaction.generate(Context())

        assert result.inputs == 0
        assert result.outputs == "empty"

    async def test_context_with_previous_interactions(self):
        """Test with context containing previous interactions."""

        def history_generator(context: Context):
            if not context.previous_interactions:
                return InteractionResult(inputs="first", outputs="initial")

            last_interaction = context.previous_interactions[-1]
            return InteractionResult(
                inputs=f"follow_up_to_{last_interaction.inputs}", outputs="response"
            )

        # First call with empty context
        interaction = DynamicInteraction(fn=history_generator)
        result1 = await interaction.generate(Context())
        assert result1.inputs == "first"
        assert result1.outputs == "initial"

        # Second call with previous interaction
        context_with_history = Context(
            previous_interactions=[
                InteractionResult(inputs="previous", outputs="result")
            ]
        )
        result2 = await interaction.generate(context_with_history)
        assert result2.inputs == "follow_up_to_previous"
        assert result2.outputs == "response"
