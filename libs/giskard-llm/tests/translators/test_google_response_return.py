"""Map Google **Interactions** API return values to :class:`ResponseResult`.

This is the Interactions (response) path. For **request** translation (``to_google``), see
``test_google_response.py``; for **generateContent** returns, see ``test_google_chat_return.py``.

``steps`` and usage: see Gemini / GenAI Interactions response shape in the
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
    FunctionCallStep,
    GoogleSearchCallStep,
    ImageContent,
    Interaction,
    ModelOutputStep,
    TextContent,
    ThoughtStep,
    Usage,
)

pytestmark = pytest.mark.google

_MODEL = "gemini-3.5-flash"
_FIXED_DT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _interaction(
    steps: list[object],
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
            steps=steps,
            usage=usage,
        )
    return Interaction.model_construct(
        id="int_test",
        created=_FIXED_DT,
        updated=_FIXED_DT,
        status="completed",
        steps=steps,
    )


def test_from_google_text_output():
    """``model_output`` text becomes a :class:`ResponseOutputMessage` with ``ResponseOutputText``."""
    raw = _interaction(
        [
            ModelOutputStep(
                type="model_output",
                content=[TextContent(type="text", text="Hello from Interactions.")],
            )
        ],
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
    raw = _interaction(
        [
            ModelOutputStep(
                type="model_output",
                content=[TextContent(type="text", text="no usage")],
            )
        ]
    )
    out = GoogleResponseTranslator.from_google(raw, _MODEL)
    assert out.usage is None


def test_from_google_function_call():
    """``function_call`` step maps to :class:`ResponseOutputFunctionCall`."""
    raw = _interaction(
        [
            FunctionCallStep(
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
    """Multiple ``steps`` list entries preserve order; each model text is a ``ResponseOutputMessage``."""
    raw = _interaction(
        [
            ModelOutputStep(
                type="model_output",
                content=[TextContent(type="text", text="A")],
            ),
            ModelOutputStep(
                type="model_output",
                content=[TextContent(type="text", text="B")],
            ),
            FunctionCallStep(
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


# -- Thought steps / unsupported content ------------------------------------------


def test_from_google_thought_step_maps_to_reasoning_item():
    """``thought`` steps become :class:`ResponseReasoningItem` and stay out of ``output_text``."""
    from giskard.llm.types import ResponseReasoningItem, ResponseReasoningSummary

    raw = _interaction(
        [
            ThoughtStep(
                type="thought",
                signature="sig-abc",
                summary=[TextContent(type="text", text="Considering the question")],
            ),
            ModelOutputStep(
                type="model_output",
                content=[TextContent(type="text", text="42")],
            ),
        ]
    )
    out = GoogleResponseTranslator.from_google(raw, _MODEL)
    assert out.outputs[0] == ResponseReasoningItem(
        id="thought_int_test_0",
        summary=[ResponseReasoningSummary(text="Considering the question")],
        encrypted_content="sig-abc",
    )
    assert out.reasoning == [out.outputs[0]]
    assert out.output_text == "42"


def _image() -> ImageContent:
    return ImageContent(type="image", data="aGVsbG8=", mime_type="image/png")


def _unsupported_cases() -> list[object]:
    return [
        pytest.param(
            GoogleSearchCallStep.model_construct(type="google_search_call", id="gs_1"),
            "google_search_call",
            id="step",
        ),
        pytest.param(
            ModelOutputStep(type="model_output", content=[_image()]),
            "model_output:image",
            id="model-output-content",
        ),
        pytest.param(
            ThoughtStep(type="thought", summary=[_image()]),
            "thought:image",
            id="thought-summary-content",
        ),
    ]


@pytest.mark.parametrize(("step", "content_type"), _unsupported_cases())
def test_from_google_unsupported_content_raises(step: object, content_type: str):
    """Steps and content parts we do not model raise instead of being silently dropped."""
    from giskard.llm.errors import UnsupportedContentError

    with pytest.raises(UnsupportedContentError) as exc_info:
        GoogleResponseTranslator.from_google(_interaction([step]), _MODEL)
    assert exc_info.value.content_type == content_type


@pytest.mark.parametrize(("step", "content_type"), _unsupported_cases())
def test_from_google_unsupported_content_dropped_with_warning(
    step: object, content_type: str, caplog: pytest.LogCaptureFixture
):
    """With ``ignore_unsupported_content`` the content is dropped and a warning logged."""
    text = ModelOutputStep(
        type="model_output", content=[TextContent(type="text", text="kept")]
    )
    with caplog.at_level("WARNING"):
        out = GoogleResponseTranslator.from_google(
            _interaction([step, text]), _MODEL, ignore_unsupported_content=True
        )
    assert out.output_text == "kept"
    assert content_type in caplog.text
