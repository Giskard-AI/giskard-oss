import pytest
from giskard.agents.chat import Chat
from giskard.llm.types import AssistantMessage, ChatMessage, UserMessage
from pydantic import BaseModel


def test_chat_output():
    """Test that Message.parse correctly parses JSON content to a Pydantic model."""

    class Cheese(BaseModel):
        name: str
        region: str

    messages: list[ChatMessage] = [
        UserMessage(content="What is the best cheese? Answer in JSON."),
        AssistantMessage(content='{"name": "Camembert", "region": "Normandy"}'),
    ]

    chat = Chat(messages=messages, output_model=Cheese)

    assert isinstance(chat.output, Cheese)
    assert chat.output == Cheese(name="Camembert", region="Normandy")


def test_chat_output_without_output_model():
    messages: list[ChatMessage] = [
        UserMessage(content="What is the best cheese? Answer in JSON."),
        AssistantMessage(content='{"name": "Camembert", "region": "Normandy"}'),
    ]

    chat = Chat(messages=messages)
    with pytest.raises(ValueError):
        chat.output
