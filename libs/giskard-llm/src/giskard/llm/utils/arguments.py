import json
from typing import Any


def serialize_arguments(arguments: dict[str, Any] | str) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def deserialize_arguments(arguments: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(arguments, str):
        return json.loads(arguments)
    return arguments
