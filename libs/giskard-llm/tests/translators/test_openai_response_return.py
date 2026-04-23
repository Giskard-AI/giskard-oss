"""Map OpenAI **Responses** API return values to :class:`ResponseResult`.

This is the ``/responses`` path (``aresp`` / ``Response``). For **request** translation
(``to_openai``), see ``test_openai_response.py``; for **Chat Completions** returns, see
``test_openai_chat_return.py``.

Output items: https://platform.openai.com/docs/api-reference/responses/object#responses/object-output
"""

import json

import pytest

pytest.importorskip("openai")

from giskard.llm.translators.openai_response import OpenAIResponseTranslator
from giskard.llm.types import ResponseOutputFunctionCall
from giskard.llm.types import ResponseOutputText as ROT
from openai.types.responses.response import Response
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

pytestmark = pytest.mark.openai

_MODEL = "gpt-4o-mini"


def _usage(input_tokens: int, output_tokens: int, total_tokens: int) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=output_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=total_tokens,
    )


def test_from_openai_output_text():
    """``message`` item with ``output_text`` becomes :class:`ResponseOutputText` entries."""
    msg = ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[
            ResponseOutputText(
                type="output_text",
                text="Hello from Responses.",
                annotations=[],
            )
        ],
    )
    raw = Response.model_construct(
        id="resp_1",
        output=[msg],
        model=_MODEL,
        usage=_usage(10, 5, 15),
    )
    out = OpenAIResponseTranslator.from_openai(raw)
    assert out.id == "resp_1"
    assert out.model == _MODEL
    assert out.usage is not None
    assert out.usage.prompt_tokens == 10
    assert out.usage.completion_tokens == 5
    assert out.usage.total_tokens == 15
    assert len(out.outputs) == 1
    assert isinstance(out.outputs[0], ROT)
    assert out.outputs[0].text == "Hello from Responses."


def test_from_openai_omit_usage():
    """``usage`` is optional on the response object."""
    msg = ResponseOutputMessage(
        id="m1",
        type="message",
        role="assistant",
        status="completed",
        content=[
            ResponseOutputText(type="output_text", text="No usage.", annotations=[])
        ],
    )
    raw = Response.model_construct(
        id="resp_2",
        output=[msg],
        model=_MODEL,
    )
    out = OpenAIResponseTranslator.from_openai(raw)
    assert out.usage is None


def test_from_openai_two_output_text_blocks_in_one_message():
    """A single message may contain several ``output_text`` blocks, each mapped separately."""
    msg = ResponseOutputMessage(
        id="m2",
        type="message",
        role="assistant",
        status="completed",
        content=[
            ResponseOutputText(type="output_text", text="First", annotations=[]),
            ResponseOutputText(type="output_text", text="Second", annotations=[]),
        ],
    )
    raw = Response.model_construct(id="resp_3", output=[msg], model=_MODEL)
    out = OpenAIResponseTranslator.from_openai(raw)
    assert len(out.outputs) == 2
    assert [o.text for o in out.outputs if isinstance(o, ROT)] == [
        "First",
        "Second",
    ]


def test_from_openai_function_call():
    """``function_call`` item maps to :class:`ResponseOutputFunctionCall` with parsed arguments."""
    fc = ResponseFunctionToolCall(
        type="function_call",
        call_id="call_abc",
        name="get_weather",
        arguments=json.dumps({"city": "Paris"}),
    )
    raw = Response.model_construct(id="resp_4", output=[fc], model=_MODEL)
    out = OpenAIResponseTranslator.from_openai(raw)
    assert len(out.outputs) == 1
    o = out.outputs[0]
    assert isinstance(o, ResponseOutputFunctionCall)
    assert o.call_id == "call_abc"
    assert o.name == "get_weather"
    assert o.arguments == {"city": "Paris"}


def test_from_openai_message_then_function():
    """Output order: assistant text then a function call, when both are present."""
    msg = ResponseOutputMessage(
        id="m3",
        type="message",
        role="assistant",
        status="completed",
        content=[
            ResponseOutputText(type="output_text", text="Calling tool…", annotations=[])
        ],
    )
    fc = ResponseFunctionToolCall(
        type="function_call",
        call_id="call_1",
        name="f",
        arguments="{}",
    )
    raw = Response.model_construct(id="resp_5", output=[msg, fc], model=_MODEL)
    out = OpenAIResponseTranslator.from_openai(raw)
    assert len(out.outputs) == 2
    assert isinstance(out.outputs[0], ROT)
    assert out.outputs[0].text == "Calling tool…"
    assert isinstance(out.outputs[1], ResponseOutputFunctionCall)
    assert out.outputs[1].name == "f"
