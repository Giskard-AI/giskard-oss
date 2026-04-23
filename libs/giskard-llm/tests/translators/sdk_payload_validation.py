"""Optional validation of translator payloads against provider SDK schemas (when installed)."""

import importlib.util

from pydantic import TypeAdapter


def validate_openai_completion_params(payload: object) -> None:
    if importlib.util.find_spec("openai") is None:
        return
    from openai.types.chat.completion_create_params import (
        CompletionCreateParamsNonStreaming,
    )

    _ = TypeAdapter(CompletionCreateParamsNonStreaming).validate_python(payload)


def validate_google_contents(contents: object) -> None:
    if importlib.util.find_spec("google.genai") is None:
        return
    from google.genai.types import Content

    assert isinstance(contents, list)
    for item in contents:
        # ``ContentListUnionDict`` allows a single part dict for ``parts``; ``Content`` expects a list.
        to_validate = item
        if isinstance(item, dict):
            parts = item.get("parts")
            if isinstance(parts, dict):
                to_validate = {**item, "parts": [parts]}
        _ = Content.model_validate(to_validate)


def validate_anthropic_message_create(payload: object) -> None:
    if importlib.util.find_spec("anthropic") is None:
        return
    from anthropic.types.message_create_params import MessageCreateParams

    _ = TypeAdapter(MessageCreateParams).validate_python(payload)


def validate_openai_response_params(payload: object) -> None:
    if importlib.util.find_spec("openai") is None:
        return
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsNonStreaming,
    )

    _ = TypeAdapter(ResponseCreateParamsNonStreaming).validate_python(payload)


def validate_google_interaction_params(payload: object) -> None:
    if importlib.util.find_spec("google.genai._interactions") is None:
        return
    from google.genai._interactions.types.interaction_create_params import (
        CreateModelInteractionParamsNonStreaming,
    )

    _ = TypeAdapter(CreateModelInteractionParamsNonStreaming).validate_python(payload)
