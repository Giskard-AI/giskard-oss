import pytest
from giskard.checks.core.interaction.interaction import ChatInteraction
from giskard.checks.core.interaction.trace import ChatTrace
from giskard.llm.types import (
    AssistantMessage,
    DeveloperMessage,
    FunctionMessage,
    SystemMessage,
    ToolCall,
    ToolCallFunction,
    ToolMessage,
    UserMessage,
)


class TestChatTraceFromMessages:
    def test_groups_messages_into_interactions(self) -> None:
        messages = [
            SystemMessage(content="sys"),
            UserMessage(content="u1"),
            AssistantMessage(
                tool_calls=[
                    ToolCall(
                        function=ToolCallFunction(
                            name="fn", arguments={"arg": "value"}
                        ),
                        id="tc_1",
                    )
                ]
            ),
            ToolMessage(content="tool-out", tool_call_id="tc_1"),
            AssistantMessage(content="a2"),
            UserMessage(content="u2"),
            AssistantMessage(content="a3"),
        ]

        trace = ChatTrace.from_messages(messages)

        assert len(trace.interactions) == 2
        first, second = trace.interactions
        assert isinstance(first, ChatInteraction)
        assert isinstance(second, ChatInteraction)

        assert trace.messages == messages

        assert first.inputs == messages[:2]
        assert first.outputs == messages[2:5]
        assert second.inputs == messages[5:6]
        assert second.outputs == messages[6:]

    def test_handles_only_inputs(self) -> None:
        trace = ChatTrace.from_messages(
            [DeveloperMessage(content="dev"), UserMessage(content="u")]
        )
        assert len(trace.interactions) == 1
        interaction = trace.interactions[0]
        assert isinstance(interaction, ChatInteraction)
        assert [m.role for m in interaction.inputs] == ["developer", "user"]
        assert interaction.outputs == []

    def test_handles_only_outputs(self) -> None:
        trace = ChatTrace.from_messages(
            [
                AssistantMessage(content="a"),
                ToolMessage(content="t", tool_call_id="tc_1"),
                FunctionMessage(content="f", name="fn"),
            ]
        )
        assert len(trace.interactions) == 1
        interaction = trace.interactions[0]
        assert isinstance(interaction, ChatInteraction)
        assert interaction.inputs == []
        assert [m.role for m in interaction.outputs] == [
            "assistant",
            "tool",
            "function",
        ]

    def test_empty_messages_creates_empty_trace(self) -> None:
        trace = ChatTrace.from_messages([])
        assert len(trace.interactions) == 0

    def test_accepts_chat_message_params(self) -> None:
        trace = ChatTrace.from_messages(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ]
        )

        assert len(trace.interactions) == 1
        interaction = trace.interactions[0]
        assert isinstance(interaction, ChatInteraction)
        assert [m.role for m in interaction.inputs] == ["user"]
        assert [m.role for m in interaction.outputs] == ["assistant"]
        assert [m.text for m in interaction.inputs] == ["hi"]
        assert [m.text for m in interaction.outputs] == ["yo"]

    def test_invalid_message_param_raises(self) -> None:
        with pytest.raises(Exception):
            _ = ChatTrace.from_messages([{"role": "user", "content": 123}])  # pyright: ignore[reportArgumentType]
