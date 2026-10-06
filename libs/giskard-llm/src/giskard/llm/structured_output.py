"""Shared structured-output JSON Schema normalization for provider translators."""

import copy
import logging
from collections.abc import Callable, Iterator
from typing import Any, Literal

from pydantic import BaseModel

from .errors import BadRequestError

SchemaMutationMode = Literal["warn", "raise", "ignore"]
StructuredOutputProfile = Literal["anthropic", "openai", "google"]

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_MUTATION: SchemaMutationMode = "warn"

_GOOGLE_JSON_SCHEMA_KEYS = frozenset(
    {
        "$id",
        "$defs",
        "$ref",
        "$anchor",
        "type",
        "format",
        "title",
        "description",
        "enum",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "anyOf",
        "oneOf",
        "properties",
        "additionalProperties",
        "required",
        "propertyOrdering",
    }
)


def coerce_schema_mutation(value: Any) -> SchemaMutationMode:
    """Validate configure-time ``schema_mutation`` values."""
    if value in ("warn", "raise", "ignore"):
        return value
    raise ValueError(
        f"schema_mutation must be one of 'warn', 'raise', or 'ignore' (got {value!r})."
    )


def pop_schema_mutation(params: dict[str, Any]) -> SchemaMutationMode:
    """Remove ``schema_mutation`` from completion params (provider-internal)."""
    raw = params.pop("schema_mutation", DEFAULT_SCHEMA_MUTATION)
    return coerce_schema_mutation(raw)


def _iter_schema_nodes(node: object) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_schema_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_schema_nodes(item)


def object_schema_paths_missing_additional_properties_false(
    schema: object, *, path: str = "$"
) -> list[str]:
    """Return JSON paths of object nodes without ``additionalProperties: false``."""
    missing: list[str] = []
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            if schema.get("additionalProperties") is not False:
                missing.append(path)
        for key, value in schema.items():
            missing.extend(
                object_schema_paths_missing_additional_properties_false(
                    value, path=f"{path}.{key}"
                )
            )
    elif isinstance(schema, list):
        for index, value in enumerate(schema):
            missing.extend(
                object_schema_paths_missing_additional_properties_false(
                    value, path=f"{path}[{index}]"
                )
            )
    return missing


def _set_additional_properties_false_recursive(schema: dict[str, Any]) -> None:
    for node in _iter_schema_nodes(schema):
        if node.get("type") == "object":
            node["additionalProperties"] = False


def _strip_ref_siblings(schema: dict[str, Any]) -> None:
    for node in _iter_schema_nodes(schema):
        if "$ref" in node:
            ref = node["$ref"]
            node.clear()
            node["$ref"] = ref


def _strip_unsupported_google_keywords(schema: dict[str, Any]) -> None:
    for node in _iter_schema_nodes(schema):
        for key in list(node.keys()):
            if key.startswith("$"):
                continue
            if key not in _GOOGLE_JSON_SCHEMA_KEYS:
                del node[key]


def _normalize_openai(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(schema)
    _set_additional_properties_false_recursive(normalized)
    return normalized


def _normalize_google(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(schema)
    _set_additional_properties_false_recursive(normalized)
    _strip_ref_siblings(normalized)
    _strip_unsupported_google_keywords(normalized)
    return normalized


def _normalize_anthropic(model: type[BaseModel]) -> dict[str, Any]:
    from anthropic import transform_schema

    return transform_schema(model)


_PROFILE_NORMALIZERS: dict[
    StructuredOutputProfile, Callable[[type[BaseModel], dict[str, Any]], dict[str, Any]]
] = {
    "openai": lambda _model, schema: _normalize_openai(schema),
    "google": lambda _model, schema: _normalize_google(schema),
    "anthropic": lambda model, _schema: _normalize_anthropic(model),
}


def normalize_pydantic_json_schema(
    model: type[BaseModel],
    *,
    profile: StructuredOutputProfile,
    provider: str,
    mode: SchemaMutationMode = DEFAULT_SCHEMA_MUTATION,
) -> dict[str, Any]:
    """Return a provider-ready JSON Schema for a Pydantic ``response_format`` model.

    Parameters
    ----
    model
        Pydantic model class passed as ``response_format``.
    profile
        Provider-specific normalization rules (``anthropic`` uses the Anthropic SDK
        ``transform_schema`` helper).
    provider
        Provider id for errors and log messages.
    mode
        ``warn`` (default): apply normalization and log when the schema changes.
        ``raise``: refuse when normalization would alter the raw Pydantic schema.
        ``ignore``: apply normalization silently.
    """
    original = model.model_json_schema()
    normalizer = _PROFILE_NORMALIZERS[profile]
    normalized = normalizer(model, original)

    if original != normalized:
        message = (
            f"Structured output schema for {model.__name__} was normalized for "
            f"{profile} compatibility (nested objects, $defs, additionalProperties, "
            "or unsupported keywords)."
        )
        if mode == "raise":
            raise BadRequestError(400, message, provider)
        if mode == "warn":
            logger.warning("%s provider: %s", provider, message)

    return normalized
