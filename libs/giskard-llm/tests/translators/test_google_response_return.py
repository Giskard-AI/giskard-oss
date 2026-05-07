"""Map Google **Interactions** API return values to :class:`ResponseResult`.

This is the Interactions (response) path. For **request** translation (``to_google``), see
``test_google_response.py``; for **generateContent** returns, see ``test_google_chat_return.py``.

``outputs`` and usage: see Gemini / GenAI Interactions response shape in the
``google.genai`` interactions types.
"""

from datetime import datetime, timezone

import pytest

pytest.importorskip("google.genai")

from giskard.llm.translators.google_response import GoogleResponseTranslator
from giskard.llm.types import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from google.genai._interactions.types import (
    FunctionCallContent,
    Interaction,
    TextContent,
    Usage,
)

pytestmark = pytest.mark.google

_MODEL = "gemini-2.0-flash"
_FIXED_DT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _interaction(
    outputs: list[object],
    *,
    usage: object | None = None,
) -> Interaction:
    """Build a minimal :class:`Interaction` with only fields ``from_google`` reads."""
    if usage is not None:
        return Interaction.model_construct(
            id="int_test",
            created=_FIXED_DT,
            updated=_FIXED_DT,
            status="completed",
            outputs=outputs,
            usage=usage,
        )
    return Interaction.model_construct(
        id="int_test",
        created=_FIXED_DT,
        updated=_FIXED_DT,
        status="completed",
        outputs=outputs,
    )


def test_from_google_text_output():
    """``type: text`` outputs become a :class:`ResponseOutputMessage` with ``ResponseOutputText``."""
    raw = _interaction(
        [TextContent(type="text", text="Hello from Interactions.")],
        usage=Usage(
            total_input_tokens=3,
            total_output_tokens=4,
            total_tokens=7,
        ),
    )
    out = GoogleResponseTranslator.from_google(raw, _MODEL)
    assert out.id == "int_test"
    assert out.model == _MODEL
    assert out.usage is not None
    assert out.usage.input_tokens == 3
    assert out.usage.output_tokens == 4
    assert out.usage.total_tokens == 7
    assert len(out.outputs) == 1
    m = out.outputs[0]
    assert isinstance(m, ResponseOutputMessage)
    assert len(m.content) == 1
    assert isinstance(m.content[0], ResponseOutputText)
    assert m.content[0].text == "Hello from Interactions."
    assert out.output_text == "Hello from Interactions."


def test_from_google_omit_usage():
    """``usage`` may be absent; ``from_google`` leaves ``ResponseResult.usage`` empty."""
    raw = _interaction([TextContent(type="text", text="no usage")])
    out = GoogleResponseTranslator.from_google(raw, _MODEL)
    assert out.usage is None


def test_from_google_function_call():
    """``function_call`` content maps to :class:`ResponseOutputFunctionCall`."""
    raw = _interaction(
        [
            FunctionCallContent(
                type="function_call",
                id="fc_1",
                name="get_weather",
                arguments={"city": "Paris"},
            )
        ],
    )
    out = GoogleResponseTranslator.from_google(raw, _MODEL)
    assert len(out.outputs) == 1
    o = out.outputs[0]
    assert isinstance(o, ResponseFunctionToolCall)
    assert o.call_id == "fc_1"
    assert o.name == "get_weather"
    assert o.arguments == {"city": "Paris"}


def test_from_google_text_then_text_then_function():
    """Multiple ``outputs`` list entries preserve order; each text is a ``ResponseOutputMessage``."""
    raw = _interaction(
        [
            TextContent(type="text", text="A"),
            TextContent(type="text", text="B"),
            FunctionCallContent(
                type="function_call",
                id="c2",
                name="f",
                arguments={"x": 1},
            ),
        ],
    )
    out = GoogleResponseTranslator.from_google(raw, _MODEL)
    assert len(out.outputs) == 3
    assert isinstance(out.outputs[0], ResponseOutputMessage)
    assert isinstance(out.outputs[0].content[0], ResponseOutputText)
    assert out.outputs[0].content[0].text == "A"
    assert isinstance(out.outputs[1], ResponseOutputMessage)
    assert isinstance(out.outputs[1].content[0], ResponseOutputText)
    assert out.outputs[1].content[0].text == "B"
    assert isinstance(out.outputs[2], ResponseFunctionToolCall)
    assert out.outputs[2].name == "f"
    assert out.output_text == "A\nB"
