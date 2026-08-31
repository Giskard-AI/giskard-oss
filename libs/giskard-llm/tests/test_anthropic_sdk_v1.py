"""Unmocked Anthropic SDK v1 checks (httpx2 http_client)."""

import pytest


async def test_sdk_v1_async_anthropic_accepts_httpx2_http_client():
    """Unmocked SDK check: v1 requires httpx2.AsyncClient for http_client."""
    pytest.importorskip("anthropic")
    pytest.importorskip("httpx2")
    import httpx2
    from anthropic import AsyncAnthropic

    http_client = httpx2.AsyncClient()
    try:
        client = AsyncAnthropic(api_key="sk-test", http_client=http_client)
        assert client is not None
    finally:
        # Caller-owned: giskard-llm does not close this; the test must.
        await http_client.aclose()
