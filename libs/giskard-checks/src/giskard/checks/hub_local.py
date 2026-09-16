"""Connect a local Python callable to Giskard Hub as a live agent.

The Hub creates an agent when this session connects and deletes it when the
WebSocket closes. Hub workers invoke the agent through a tenant-scoped call
bridge; this client receives those calls and runs ``handler``.
"""

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

from giskard.checks.utils.optional_deps import require_optional_dependency

type LocalAgentHandler = Callable[
    [dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]
]


def build_local_agent_connect_url(
    hub_url: str,
    *,
    name: str = "Local agent",
    project_id: str | None = None,
    description: str | None = None,
) -> str:
    """Build the Hub WebSocket URL used to register a local agent.

    Parameters
    ----------
    hub_url : str
        Hub origin (``https://app.example.com``) or a full ``ws(s)://`` URL.
    name : str, optional
        Display name created in Hub on connect.
    project_id : str, optional
        Project that should own the agent. Hub picks the first writable
        project when omitted.
    description : str, optional
        Optional agent description.

    Returns
    -------
    str
        WebSocket URL including query parameters.
    """
    parsed = urlparse(hub_url)
    if parsed.scheme in {"ws", "wss"} and parsed.path.rstrip("/"):
        return hub_url
    scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
    query = urlencode(
        {
            key: value
            for key, value in {
                "name": name,
                "project_id": project_id,
                "description": description,
            }.items()
            if value
        }
    )
    return urlunparse(
        (scheme, parsed.netloc, "/_api/v2/local-agents/connect", "", query, "")
    )


async def invoke_local_agent_handler(
    handler: LocalAgentHandler, payload: dict[str, Any]
) -> dict[str, Any]:
    """Run a local agent handler and require a JSON-object result.

    Parameters
    ----------
    handler : LocalAgentHandler
        Sync or async callable that receives the Hub request body.
    payload : dict
        JSON object posted by Hub (playground, eval, scan, or ART).

    Returns
    -------
    dict
        JSON object sent back to Hub as the agent output.

    Raises
    ------
    TypeError
        If the handler does not return a dict.
    """
    result = handler(payload)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, dict):
        raise TypeError("Local agent handler must return a dict")
    return result


async def handle_hub_local_agent_message(
    message: dict[str, Any],
    handler: LocalAgentHandler,
) -> dict[str, Any] | None:
    """Translate one Hub WebSocket message into a reply, if needed.

    Parameters
    ----------
    message : dict
        Decoded JSON message from Hub.
    handler : LocalAgentHandler
        Local callable that serves agent invocations.

    Returns
    -------
    dict or None
        Reply to send, or ``None`` when the message is informational.
    """
    msg_type = message.get("type")
    if msg_type == "connected":
        return None
    if msg_type == "error":
        raise RuntimeError(str(message.get("message") or "Hub local-agent error"))
    if msg_type != "call":
        return None

    call_id = message.get("id")
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return {
            "type": "error",
            "id": call_id,
            "message": "Hub call payload must be a JSON object.",
        }
    try:
        output = await invoke_local_agent_handler(handler, payload)
    except Exception as exc:
        return {"type": "error", "id": call_id, "message": str(exc)}
    return {"type": "result", "id": call_id, "output": output}


async def connect_local_agent(
    *,
    hub_url: str,
    api_key: str,
    handler: LocalAgentHandler,
    name: str = "Local agent",
    project_id: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Connect ``handler`` to Hub until the session is closed.

    Hub creates a read-only local agent on connect and deletes it on
    disconnect. Evaluations and other workers in the same tenant reach the
    handler through that live session.

    Parameters
    ----------
    hub_url : str
        Hub origin, for example ``https://app.llm.localhost``.
    api_key : str
        Tenant API key (sent as ``X-API-Key``).
    handler : LocalAgentHandler
        Sync or async callable ``payload -> dict``. For chat agents, Hub
        sends ``{"messages": [...]}`` and expects
        ``{"response": {"role": "assistant", "content": "..."}}``.
    name : str, optional
        Agent name shown in Hub.
    project_id : str, optional
        Project UUID. When omitted, Hub uses the first project the key can
        create agents in.
    description : str, optional
        Optional description stored on the Hub agent.

    Returns
    -------
    dict
        The Hub ``connected`` handshake (includes ``agent_id`` and
        ``project_id``) after the session ends.

    Example
    -------
    Install with ``pip install 'giskard-checks[hub]'``, then::

        async def echo(payload: dict) -> dict:
            text = payload["messages"][-1]["content"]
            return {"response": {"role": "assistant", "content": text}}

        await connect_local_agent(
            hub_url="https://app.llm.localhost",
            api_key="...",
            handler=echo,
        )
    """
    websockets = require_optional_dependency(
        "websockets",
        install_hint="Install giskard-checks[hub] to connect a local agent to the Hub.",
    )
    uri = build_local_agent_connect_url(
        hub_url, name=name, project_id=project_id, description=description
    )
    handshake: dict[str, Any] | None = None
    async with websockets.connect(
        uri, additional_headers={"X-API-Key": api_key}
    ) as websocket:
        raw = await websocket.recv()
        message = json.loads(raw)
        if message.get("type") == "error":
            raise RuntimeError(
                str(message.get("message") or "Hub rejected the local agent connection")
            )
        if message.get("type") != "connected":
            raise RuntimeError(f"Unexpected Hub handshake: {message!r}")
        handshake = message
        async for raw in websocket:
            reply = await handle_hub_local_agent_message(json.loads(raw), handler)
            if reply is not None:
                await websocket.send(json.dumps(reply))
    return handshake or {}
