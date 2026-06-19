import copy
import hashlib
import json
from collections.abc import AsyncGenerator
from typing import Any, override

from pydantic import BaseModel, Field, model_validator

from ..core import Trace
from ..core.input_generator import InputGenerator
from ..core.mixin import WithGeneratorMixin

PROMPT_PLACEHOLDER = "{{prompt}}"
_MAPPING_TEMPLATE = "giskard.checks::generators/dataset_input_mapping.j2"


class MappingTemplate[T](BaseModel):  # pyright: ignore[reportMissingTypeArgument]
    """LLM output: a valid instance of the target schema with a {{prompt}} marker.

    Either ``message`` (a valid ``T`` containing the ``{{prompt}}`` token in the
    string field(s) that should carry the user's message) or ``schema_issue``
    (set when no string field can hold a message) is provided, never both.
    """

    schema_issue: str | None = Field(
        default=None,
        description=(
            "Schema issue preventing templating (e.g. no string-like field to "
            "carry a user message). Set this instead of message when the schema "
            "cannot hold a user prompt."
        ),
    )
    message: T | None = Field(
        default=None,
        description=(
            "A valid instance of the target schema with the literal token "
            "'{{prompt}}' placed in the string field(s) that carry the user's "
            "message. May embed it in surrounding text, e.g. 'Please answer to "
            "{{prompt}}' or just '{{prompt}}'. Other required fields get neutral "
            "placeholder values. None when schema_issue is set."
        ),
    )

    @model_validator(mode="after")
    def _xor(self) -> "MappingTemplate[T]":
        if (self.message is None) == (self.schema_issue is None):
            raise ValueError("Exactly one of 'message' / 'schema_issue' must be set")
        return self


def _replace(value: Any, prompt: str) -> tuple[Any, bool]:
    """Return ``(new_value, replaced)`` where ``replaced`` is True iff a placeholder was hit."""
    if isinstance(value, str):
        if PROMPT_PLACEHOLDER in value:
            return value.replace(PROMPT_PLACEHOLDER, prompt), True
        return value, False
    if isinstance(value, BaseModel):
        replaced = False
        data: dict[str, Any] = {}
        for name in type(value).model_fields:
            data[name], hit = _replace(getattr(value, name), prompt)
            replaced |= hit
        return type(value).model_validate(data), replaced
    if isinstance(value, list):
        replaced = False
        items: list[Any] = []
        for item in value:
            new_item, hit = _replace(item, prompt)
            items.append(new_item)
            replaced |= hit
        return items, replaced
    if isinstance(value, dict):
        # Recurse into values only; dict keys and tuple elements are
        # intentionally out of scope (JSON-shaped agent inputs never carry the
        # message in a key or a tuple). A placeholder hiding only there is not
        # substituted, and substitute_prompt then raises — fails loud, not silent.
        replaced = False
        out: dict[Any, Any] = {}
        for k, v in value.items():
            out[k], hit = _replace(v, prompt)
            replaced |= hit
        return out, replaced
    return value, False


def substitute_prompt(message: Any, prompt: str) -> Any:
    """Return a deep copy of ``message`` with every ``{{prompt}}`` replaced by ``prompt``.

    Raises ``ValueError`` if the placeholder is not present anywhere.
    """
    result, replaced = _replace(copy.deepcopy(message), prompt)
    if not replaced:
        raise ValueError(
            f"Template message contains no '{PROMPT_PLACEHOLDER}' placeholder to inject the prompt into"
        )
    return result


_TEMPLATE_CACHE: dict[str, "MappingTemplate[Any]"] = {}


def schema_cache_key(input_type: type) -> str:
    """Stable cache key: qualified class name + hash of its JSON schema."""
    schema = input_type.model_json_schema()  # type: ignore[attr-defined]
    return _schema_cache_key(input_type, schema)


def _schema_cache_key(input_type: type, schema: dict[str, Any]) -> str:
    canonical = json.dumps(schema, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{input_type.__qualname__}:{digest}"


def _cache_clear() -> None:
    """Test hook: clear the process-level template cache."""
    _TEMPLATE_CACHE.clear()


@InputGenerator.register("dataset_input")
class DatasetInputGenerator[TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    InputGenerator[TraceType], WithGeneratorMixin
):
    """Single-shot generator that places a fixed dataset prompt into the target input.

    For a ``str`` target the prompt is yielded verbatim (no LLM). For a structured
    target the prompt is injected into an LLM-resolved, schema-only template; the
    LLM never sees the prompt itself.
    """

    prompt: str = Field(
        ..., min_length=1, description="Fixed dataset prompt, used verbatim."
    )

    @override
    async def __call__(
        self, trace: TraceType, input_type: type[Any] | None = None
    ) -> AsyncGenerator[Any, TraceType]:
        T = input_type or str
        if T is str:
            yield self.prompt
            return
        template = await self._resolve_template(T)
        assert template.message is not None  # _resolve_template raises otherwise
        yield substitute_prompt(template.message, self.prompt)

    async def _resolve_template(self, input_type: type) -> "MappingTemplate[Any]":
        schema_dict = input_type.model_json_schema()  # type: ignore[attr-defined]
        key = _schema_cache_key(input_type, schema_dict)
        cached = _TEMPLATE_CACHE.get(key)
        if cached is not None:
            return cached

        schema = json.dumps(schema_dict, indent=2, default=str)
        workflow = self._generator.template(_MAPPING_TEMPLATE).with_output(
            MappingTemplate[input_type]
        )
        result = await workflow.with_inputs(schema=schema).run()
        template = result.output
        if template.schema_issue is not None:
            raise ValueError(
                f"Cannot template prompt into {input_type.__qualname__}: {template.schema_issue}"
            )
        _TEMPLATE_CACHE[key] = template
        return template
