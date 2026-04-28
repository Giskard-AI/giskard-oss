"""Map OpenAI **Chat Completions** return values to :class:`CompletionResponse`.

This is the ``acompletion`` / :class:`ChatCompletion` path. For the **Responses** API: request
``to_openai`` in ``test_openai_response.py``; return ``Response`` -> ``ResponseResult`` in
``test_openai_response_return.py``.

Assistant ``message`` fields: https://platform.openai.com/docs/api-reference/chat/object
"""

import json

import pytest

pytest.importorskip("openai")

from giskard.llm.translators.openai_chat import OpenAIChatTranslator
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

pytestmark = pytest.mark.openai

_MODEL = "gpt-4o-mini"


def test_from_openai_assistant_text_message():
    """Maps ``choices[].message`` with string ``content`` (typical text reply)."""
    raw = ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Hello, world.",
                ),
            )
        ],
        created=0,
        model=_MODEL,
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )
    out = OpenAIChatTranslator.from_openai(raw)
    assert out.model == _MODEL
    assert out.usage is not None
    assert out.usage.input_tokens == 10
    assert out.usage.output_tokens == 5
    assert out.usage.total_tokens == 15
    assert len(out.choices) == 1
    ch = out.choices[0]
    assert ch.index == 0
    assert ch.finish_reason == "stop"
    assert ch.message.role == "assistant"
    assert ch.message.content == "Hello, world."
    assert ch.message.refusal is None
    assert ch.message.tool_calls is None


def test_from_openai_assistant_message_omit_usage():
    """``usage`` may be absent on the completion object."""
    raw = ChatCompletion(
        id="chatcmpl-test2",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="No usage.",
                ),
            )
        ],
        created=0,
        model=_MODEL,
        object="chat.completion",
    )
    out = OpenAIChatTranslator.from_openai(raw)
    assert out.usage is None
    assert out.choices[0].message.content == "No usage."


def test_from_openai_assistant_refusal():
    """Maps ``message.refusal`` when the model returns a policy refusal (no ``content``)."""
    raw = ChatCompletion(
        id="chatcmpl-refusal",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    refusal="I'm sorry, I can't assist with that.",
                ),
            )
        ],
        created=0,
        model=_MODEL,
        object="chat.completion",
    )
    out = OpenAIChatTranslator.from_openai(raw)
    msg = out.choices[0].message
    assert msg.content is None
    assert msg.refusal == "I'm sorry, I can't assist with that."


def test_from_openai_assistant_text_and_tool_calls():
    """Assistant `message` may include both `content` and `tool_calls` (e.g. a short preamble)."""
    raw = ChatCompletion(
        id="chatcmpl-tools",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="I will use the function.",
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id="call_abc",
                            type="function",
                            function=Function(
                                name="get_weather",
                                arguments=json.dumps({"city": "Paris"}),
                            ),
                        )
                    ],
                ),
            )
        ],
        created=0,
        model=_MODEL,
        object="chat.completion",
    )
    out = OpenAIChatTranslator.from_openai(raw)
    msg = out.choices[0].message
    assert msg.content == "I will use the function."
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "call_abc"
    assert msg.tool_calls[0].type == "function"
    assert msg.tool_calls[0].function.name == "get_weather"
    assert msg.tool_calls[0].function.arguments == {"city": "Paris"}
    assert out.choices[0].finish_reason == "tool_calls"


def test_from_openai_assistant_tool_calls_only():
    """Tool-only turn: `content` is null, `tool_calls` populated, `finish_reason` is `tool_calls`."""
    raw = ChatCompletion(
        id="chatcmpl-toolonly",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id="call_1",
                            type="function",
                            function=Function(
                                name="f",
                                arguments=json.dumps({"x": 1}),
                            ),
                        )
                    ],
                ),
            )
        ],
        created=0,
        model=_MODEL,
        object="chat.completion",
    )
    out = OpenAIChatTranslator.from_openai(raw)
    msg = out.choices[0].message
    assert msg.content is None
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].function.arguments == {"x": 1}


def test_from_openai_two_choices():
    """Maps multiple `choices` when `n` > 1 (each index and message preserved)."""
    raw = ChatCompletion(
        id="chatcmpl-n2",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="A"),
            ),
            Choice(
                index=1,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="B"),
            ),
        ],
        created=0,
        model=_MODEL,
        object="chat.completion",
    )
    out = OpenAIChatTranslator.from_openai(raw)
    assert len(out.choices) == 2
    assert out.choices[0].index == 0
    assert out.choices[0].message.content == "A"
    assert out.choices[1].index == 1
    assert out.choices[1].message.content == "B"
