import json
from typing import Any


def serialize_arguments(arguments: dict[str, Any] | str) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def deserialize_arguments(arguments: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool arguments: {e}") from e
    return arguments
