"""Optional validation of translator payloads against provider SDK schemas (when installed)."""

import importlib
import importlib.util
from typing import Any

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


def validate_google_generate_content_config(config: object) -> None:
    if importlib.util.find_spec("google.genai") is None:
        return
    from google.genai.types import GenerateContentConfig

    _ = GenerateContentConfig.model_validate(config)


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


def google_interaction_types() -> Any:
    """Module holding the Interactions API types of the installed google-genai.

    google-genai 3.x moved them from the private ``_interactions.types`` package
    to ``_gaos.types.interactions``; 2.x only has the former. Returns ``None``
    when neither exists (SDK not installed).
    """
    for name in (
        "google.genai._gaos.types.interactions",
        "google.genai._interactions.types",
    ):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def validate_google_interaction_params(payload: object) -> None:
    types = google_interaction_types()
    if types is None:
        return
    if hasattr(types, "CreateModelInteraction"):  # google-genai >= 3 (pydantic model)
        params_type = types.CreateModelInteraction
    else:
        params_type = importlib.import_module(
            "google.genai._interactions.types.interaction_create_params"
        ).CreateModelInteractionParamsNonStreaming

    _ = TypeAdapter(params_type).validate_python(payload)
