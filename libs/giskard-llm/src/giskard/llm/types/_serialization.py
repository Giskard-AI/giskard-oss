from collections import defaultdict
from typing import Any, Callable, cast

from pydantic import BaseModel, SerializationInfo

_SERIALIZER_DICT: dict[str, dict[type[BaseModel], Callable[..., Any]]] = defaultdict(
    dict
)


def _register_serializer[T: BaseModel](
    provider: str, model: type[T], serializer: Callable[..., Any]
):
    _SERIALIZER_DICT[provider][model] = serializer


def register_serializer[T: BaseModel](model: type[T], provider: str):
    """
    Register a serializer for *provider* keyed by the first parameter's type hint.

    The decorated callable must annotate its first parameter with a concrete
    ``BaseModel`` subclass (second parameter is typically ``SerializationInfo``).
    """

    def decorator(
        func: Callable[[T, SerializationInfo], Any],
    ) -> Callable[[T, SerializationInfo], Any]:
        _register_serializer(provider, model, func)
        return func

    return decorator


def get_serializer[T: BaseModel](
    model: type[T], provider: str
) -> Callable[[T, SerializationInfo], Any] | None:
    if model not in _SERIALIZER_DICT[provider]:
        return None
    return cast(
        Callable[[T, SerializationInfo], Any], _SERIALIZER_DICT[provider][model]
    )


def close_object_schemas(schema: Any) -> Any:
    """Close object schemas without rewriting literal instance values.

    Parameters
    ----------
    schema : Any
        Schema or sequence of schemas to update in place.

    Returns
    -------
    Any
        The input, with object schemas closed unless they explicitly allow keys.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema.setdefault("additionalProperties", False)
        for keyword in (
            "$defs",
            "definitions",
            "properties",
            "patternProperties",
            "dependentSchemas",
            "dependencies",
        ):
            subschemas = schema.get(keyword)
            if isinstance(subschemas, dict):
                for value in subschemas.values():
                    close_object_schemas(value)
        for keyword in (
            "additionalProperties",
            "unevaluatedProperties",
            "propertyNames",
            "items",
            "additionalItems",
            "contains",
            "unevaluatedItems",
            "contentSchema",
            "allOf",
            "anyOf",
            "oneOf",
            "prefixItems",
            "not",
            "if",
            "then",
            "else",
        ):
            close_object_schemas(schema.get(keyword))
    elif isinstance(schema, list):
        for value in schema:
            close_object_schemas(value)
    return schema
