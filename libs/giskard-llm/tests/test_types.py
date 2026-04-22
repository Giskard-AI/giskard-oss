from giskard.llm.types import (
    Choice,
    ChoiceMessage,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    ToolCall,
    ToolCallFunction,
)


def test_completion_response_model_dump():
    resp = CompletionResponse(
        choices=[
            Choice(
                message=ChoiceMessage(role="assistant", content="Hello"),
                finish_reason="stop",
            )
        ],
        model="gpt-4o",
    )
    dump = resp.model_dump()
    assert dump["choices"][0]["message"]["role"] == "assistant"
    assert dump["choices"][0]["message"]["content"] == "Hello"
    assert dump["choices"][0]["finish_reason"] == "stop"
    assert dump["model"] == "gpt-4o"


def test_choice_message_excludes_none():
    msg = ChoiceMessage(role="assistant", content="Hello")
    dump = msg.model_dump()
    assert "tool_calls" not in dump


def test_choice_message_includes_typed_tool_calls():
    msg = ChoiceMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_1",
                type="function",
                function=ToolCallFunction(name="add", arguments={"a": 1, "b": 2}),
            )
        ],
    )
    dump = msg.model_dump()
    assert dump["tool_calls"] is not None
    assert len(dump["tool_calls"]) == 1
    assert dump["tool_calls"][0]["function"]["name"] == "add"


def test_tool_call_model():
    tc = ToolCall(
        id="call_1",
        function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"}),
    )
    assert tc.id == "call_1"
    assert tc.type == "function"
    assert tc.function.name == "get_weather"


def test_embedding_response():
    resp = EmbeddingResponse(
        data=[
            EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0),
            EmbeddingData(embedding=[0.4, 0.5, 0.6], index=1),
        ]
    )
    assert len(resp.data) == 2
    assert resp.data[0].embedding == [0.1, 0.2, 0.3]
