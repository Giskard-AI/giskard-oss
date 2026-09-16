import json
from unittest.mock import MagicMock, patch

import pytest
from giskard.checks.hub_local import (
    build_local_agent_connect_url,
    connect_local_agent,
    handle_hub_local_agent_message,
    invoke_local_agent_handler,
)


def test_build_url_from_https_origin():
    url = build_local_agent_connect_url(
        "https://app.llm.localhost",
        name="Demo",
        project_id="proj-1",
    )
    assert url.startswith("wss://app.llm.localhost/_api/v2/local-agents/connect")
    assert "name=Demo" in url
    assert "project_id=proj-1" in url


def test_build_url_from_http_origin():
    url = build_local_agent_connect_url("http://localhost:8000")
    assert url.startswith("ws://localhost:8000/_api/v2/local-agents/connect")


def test_build_url_keeps_explicit_ws_path():
    raw = "wss://hub.example/_api/v2/local-agents/connect?name=x"
    assert build_local_agent_connect_url(raw) == raw


@pytest.mark.asyncio
async def test_sync_and_async_handlers():
    async def async_handler(payload: dict) -> dict:
        return {"echo": payload["n"]}

    assert await invoke_local_agent_handler(lambda payload: {"n": 1}, {"n": 0}) == {
        "n": 1
    }
    assert await invoke_local_agent_handler(async_handler, {"n": 2}) == {"echo": 2}


@pytest.mark.asyncio
async def test_handler_must_return_dict():
    with pytest.raises(TypeError, match="must return a dict"):
        await invoke_local_agent_handler(lambda payload: "nope", {})


@pytest.mark.asyncio
async def test_call_message_runs_handler():
    reply = await handle_hub_local_agent_message(
        {"type": "call", "id": "abc", "payload": {"messages": [{"content": "hi"}]}},
        lambda payload: {
            "response": {
                "role": "assistant",
                "content": payload["messages"][0]["content"],
            }
        },
    )
    assert reply == {
        "type": "result",
        "id": "abc",
        "output": {"response": {"role": "assistant", "content": "hi"}},
    }


@pytest.mark.asyncio
async def test_call_message_reports_handler_error():
    def boom(_payload: dict) -> dict:
        raise RuntimeError("nope")

    reply = await handle_hub_local_agent_message(
        {"type": "call", "id": "abc", "payload": {}}, boom
    )
    assert reply == {"type": "error", "id": "abc", "message": "nope"}


@pytest.mark.asyncio
async def test_hub_error_message_raises():
    with pytest.raises(RuntimeError, match="denied"):
        await handle_hub_local_agent_message(
            {"type": "error", "message": "denied"}, lambda payload: {}
        )


@pytest.mark.asyncio
async def test_connect_local_agent_roundtrip():
    incoming = [
        json.dumps({"type": "connected", "agent_id": "a1", "project_id": "p1"}),
        json.dumps(
            {"type": "call", "id": "c1", "payload": {"messages": [{"content": "ping"}]}}
        ),
    ]

    class FakeWs:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self._incoming = list(incoming)

        async def recv(self) -> str:
            return self._incoming.pop(0)

        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            if not self._incoming:
                raise StopAsyncIteration
            return self._incoming.pop(0)

        async def send(self, data: str) -> None:
            self.sent.append(data)

    fake = FakeWs()

    class _CM:
        async def __aenter__(self):
            return fake

        async def __aexit__(self, exc_type, exc, tb):
            return False

    fake_ws_module = MagicMock()
    fake_ws_module.connect.return_value = _CM()

    with patch(
        "giskard.checks.hub_local.require_optional_dependency",
        return_value=fake_ws_module,
    ):
        hello = await connect_local_agent(
            hub_url="https://app.llm.localhost",
            api_key="secret",
            handler=lambda payload: {
                "response": {
                    "role": "assistant",
                    "content": payload["messages"][0]["content"],
                }
            },
            name="Echo",
        )

    assert hello["agent_id"] == "a1"
    assert json.loads(fake.sent[0])["output"]["response"]["content"] == "ping"
    fake_ws_module.connect.assert_called_once()
    kwargs = fake_ws_module.connect.call_args.kwargs
    assert kwargs["additional_headers"]["X-API-Key"] == "secret"
