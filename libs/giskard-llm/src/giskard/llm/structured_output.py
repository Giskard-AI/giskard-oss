"""Shared structured-output JSON Schema normalization for provider translators."""

import copy
import logging
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from .errors import BadRequestError

SchemaMutationMode = Literal["warn", "raise", "ignore"]
StructuredOutputProfile = Literal["anthropic", "openai", "google"]

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_MUTATION: SchemaMutationMode = "warn"

# Keywords removed only for provider/API compatibility; Pydantic still applies
# defaults and parses the same payload when these are absent from the sent schema.
_METADATA_ONLY_KEYWORDS = frozenset(
    {
        "default",
        "title",
        "description",
        "examples",
        "deprecated",
        "$comment",
        "$schema",
    }
)

# Gemini ``response_json_schema`` accepts this subset of JSON Schema keywords
# (see ``GenerateContentConfig.response_json_schema`` in google-genai). Keys
# in ``properties``, ``$defs``, ``patternProperties``, etc. are user-defined
# names and must never be filtered.
_GOOGLE_JSON_SCHEMA_KEYWORDS = frozenset(
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

_SCHEMA_NAME_MAP_KEYS = frozenset(
    {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
)

_warned_lossy_mutations: set[tuple[str, str, StructuredOutputProfile]] = set()
_warn_lock = threading.Lock()


@dataclass(frozen=True)
class SchemaKeywordChange:
    """One keyword-level schema edit from normalization."""

    path: str
    keyword: str
    action: Literal["removed", "added", "modified"]


def coerce_schema_mutation(value: Any) -> SchemaMutationMode:
    """Validate configure-time ``schema_mutation`` values."""
    if value in ("warn", "raise", "ignore"):
        return value
    raise ValueError(
        f"schema_mutation must be one of 'warn', 'raise', or 'ignore' (got {value!r})."
    )


def pop_schema_mutation(params: dict[str, Any]) -> SchemaMutationMode:
    """Remove internal ``schema_mutation`` set by provider translators (not public API)."""
    raw = params.pop("schema_mutation", DEFAULT_SCHEMA_MUTATION)
    return coerce_schema_mutation(raw)


def reject_user_schema_mutation_param(params: dict[str, Any]) -> None:
    """Drop ``schema_mutation`` if passed on ``acompletion`` / ``aresponse`` (configure-time only)."""
    if "schema_mutation" in params:
        logger.warning(
            "schema_mutation is configure-time only (LLMClient.configure); "
            "ignoring completion/response param"
        )
        params.pop("schema_mutation")


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


def _is_lossy_change(change: SchemaKeywordChange) -> bool:
    if change.action == "added" and change.keyword == "additionalProperties":
        return False
    if change.action == "removed" and change.keyword in _METADATA_ONLY_KEYWORDS:
        return False
    if change.action == "removed" and change.keyword == "$ref":
        return False
    if change.action == "modified" and change.keyword in _METADATA_ONLY_KEYWORDS:
        return False
    return True


def _compare_schema_mutation(
    original: object,
    normalized: object,
    *,
    path: str,
    changes: list[SchemaKeywordChange],
) -> None:
    if isinstance(original, dict) and isinstance(normalized, dict):
        for key, orig_val in original.items():
            if key in _SCHEMA_NAME_MAP_KEYS:
                if not isinstance(orig_val, dict):
                    continue
                norm_map = normalized.get(key)
                if not isinstance(norm_map, dict):
                    for name in orig_val:
                        changes.append(
                            SchemaKeywordChange(f"{path}.{key}.{name}", name, "removed")
                        )
                    continue
                for name, orig_child in orig_val.items():
                    if name not in norm_map:
                        changes.append(
                            SchemaKeywordChange(f"{path}.{key}.{name}", name, "removed")
                        )
                    else:
                        _compare_schema_mutation(
                            orig_child,
                            norm_map[name],
                            path=f"{path}.{key}.{name}",
                            changes=changes,
                        )
                continue

            if key not in normalized:
                if key == "$ref" and any(
                    k in normalized for k in ("type", "properties", "items", "anyOf")
                ):
                    continue
                changes.append(SchemaKeywordChange(path, key, "removed"))
                continue

            norm_val = normalized[key]
            if orig_val == norm_val:
                continue
            if key == "additionalProperties" and norm_val is False:
                changes.append(SchemaKeywordChange(path, key, "added"))
                continue
            if isinstance(orig_val, dict) and isinstance(norm_val, dict):
                _compare_schema_mutation(
                    orig_val, norm_val, path=f"{path}.{key}", changes=changes
                )
            else:
                changes.append(SchemaKeywordChange(path, key, "modified"))
        return

    if original != normalized and path != "$":
        changes.append(SchemaKeywordChange(path, "<value>", "modified"))


def analyze_schema_mutation(
    original: dict[str, Any],
    normalized: dict[str, Any],
    profile: StructuredOutputProfile,
) -> list[SchemaKeywordChange]:
    """Return keyword-level edits between Pydantic JSON Schema and normalized output."""
    del profile
    changes: list[SchemaKeywordChange] = []
    _compare_schema_mutation(original, normalized, path="$", changes=changes)
    return changes


def is_lossy_schema_mutation(
    original: dict[str, Any],
    normalized: dict[str, Any],
    profile: StructuredOutputProfile,
) -> bool:
    """True when normalization removes or alters validation-relevant schema keywords."""
    return any(
        _is_lossy_change(c)
        for c in analyze_schema_mutation(original, normalized, profile)
    )


def _format_lossy_changes(changes: list[SchemaKeywordChange]) -> str:
    lossy = [c for c in changes if _is_lossy_change(c)]
    if not lossy:
        return ""
    parts = [
        f"{change.action} {change.path} ({change.keyword})" for change in lossy[:12]
    ]
    suffix = f" (+{len(lossy) - 12} more)" if len(lossy) > 12 else ""
    return "; ".join(parts) + suffix


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
    """Strip keywords Gemini rejects. Property / ``$defs`` names are never removed."""

    def visit(node: dict[str, Any], *, in_name_map: bool) -> None:
        if in_name_map:
            for value in node.values():
                if isinstance(value, dict):
                    visit(value, in_name_map=False)
            return

        for key in list(node.keys()):
            if key in _SCHEMA_NAME_MAP_KEYS:
                sub = node[key]
                if isinstance(sub, dict):
                    visit(sub, in_name_map=True)
                continue
            if key.startswith("$"):
                continue
            if key not in _GOOGLE_JSON_SCHEMA_KEYWORDS:
                del node[key]

        for key, value in list(node.items()):
            if key in _SCHEMA_NAME_MAP_KEYS:
                continue
            if isinstance(value, dict):
                visit(value, in_name_map=False)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        visit(item, in_name_map=False)

    visit(schema, in_name_map=False)


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


def _apply_schema_mutation_policy(
    *,
    label: str,
    profile: StructuredOutputProfile,
    provider: str,
    mode: SchemaMutationMode,
    original: dict[str, Any],
    normalized: dict[str, Any],
    warn_key: tuple[str, str, StructuredOutputProfile],
) -> None:
    changes = analyze_schema_mutation(original, normalized, profile)
    if not any(_is_lossy_change(c) for c in changes):
        return
    detail = _format_lossy_changes(changes)
    message = (
        f"Structured output schema for {label} was lossily normalized for {profile} "
        f"compatibility: {detail}"
    )
    if mode == "raise":
        raise BadRequestError(400, message, provider)
    if mode == "warn":
        with _warn_lock:
            is_new = warn_key not in _warned_lossy_mutations
            if is_new:
                _warned_lossy_mutations.add(warn_key)
        if is_new:
            logger.warning("%s provider: %s", provider, message)


_PROFILE_NORMALIZERS: dict[
    StructuredOutputProfile, Callable[[type[BaseModel], dict[str, Any]], dict[str, Any]]
] = {
    "openai": lambda _model, schema: _normalize_openai(schema),
    "google": lambda _model, schema: _normalize_google(schema),
    "anthropic": lambda model, _schema: _normalize_anthropic(model),
}


def normalize_json_schema(
    schema: dict[str, Any],
    *,
    profile: StructuredOutputProfile,
    provider: str,
    mode: SchemaMutationMode = DEFAULT_SCHEMA_MUTATION,
    schema_label: str = "schema",
) -> dict[str, Any]:
    """Normalize a raw JSON Schema dict (non-Anthropic profiles only)."""
    if profile == "anthropic":
        raise ValueError(
            "normalize_json_schema does not support profile='anthropic'; use a Pydantic model"
        )
    original = copy.deepcopy(schema)
    if profile == "openai":
        normalized = _normalize_openai(original)
    else:
        normalized = _normalize_google(original)
    warn_key = ("", schema_label, profile)
    _apply_schema_mutation_policy(
        label=schema_label,
        profile=profile,
        provider=provider,
        mode=mode,
        original=original,
        normalized=normalized,
        warn_key=warn_key,
    )
    return normalized


def normalize_pydantic_json_schema(
    model: type[BaseModel],
    *,
    profile: StructuredOutputProfile,
    provider: str,
    mode: SchemaMutationMode = DEFAULT_SCHEMA_MUTATION,
) -> dict[str, Any]:
    """Return a provider-ready JSON Schema for a Pydantic ``response_format`` model.

    **Lossy normalization** means validation-relevant keywords or object properties
    were removed or changed (constraints, ``pattern``, ``format``, ``enum``,
    ``$ref`` siblings, missing ``properties`` / ``$defs`` entries, etc.). It does
    **not** include metadata-only removals (``default``, ``title``, ``description``,
    …) or adding ``additionalProperties: false`` alone.
    """
    original = model.model_json_schema()
    normalizer = _PROFILE_NORMALIZERS[profile]
    normalized = normalizer(model, original)
    warn_key = (model.__module__, model.__qualname__, profile)
    _apply_schema_mutation_policy(
        label=model.__name__,
        profile=profile,
        provider=provider,
        mode=mode,
        original=original,
        normalized=normalized,
        warn_key=warn_key,
    )
    return normalized


def reset_schema_mutation_warnings_for_tests() -> None:
    """Clear dedupe state (for unit tests only)."""
    _warned_lossy_mutations.clear()
