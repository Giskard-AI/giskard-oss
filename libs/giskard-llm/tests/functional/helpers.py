"""Helpers shared by functional provider tests."""

import os


def azure_foundry_v1_base_url() -> str | None:
    """Return the OpenAI-compatible Azure Foundry v1 base URL.

    The CI environment already exposes ``AZURE_AI_ENDPOINT`` for Azure AI
    Foundry tests. The OpenAI SDK expects the v1 base URL, so append
    ``/openai/v1/`` unless the caller supplied an explicit v1 endpoint.
    """
    explicit = os.getenv("AZURE_AI_OPENAI_V1_ENDPOINT")
    if explicit:
        return explicit

    endpoint = os.getenv("AZURE_AI_ENDPOINT")
    if not endpoint:
        return None

    normalized = endpoint.rstrip("/")
    if normalized.endswith("/openai/v1"):
        return f"{normalized}/"
    return f"{normalized}/openai/v1/"
