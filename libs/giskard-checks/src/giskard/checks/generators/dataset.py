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


def _replace(value: Any, prompt: str, replaced: list[bool]) -> Any:
    if isinstance(value, str):
        if PROMPT_PLACEHOLDER in value:
            replaced[0] = True
            return value.replace(PROMPT_PLACEHOLDER, prompt)
        return value
    if isinstance(value, BaseModel):
        data = {
            name: _replace(getattr(value, name), prompt, replaced)
            for name in type(value).model_fields
        }
        return type(value).model_validate(data)
    if isinstance(value, list):
        return [_replace(item, prompt, replaced) for item in value]
    if isinstance(value, dict):
        return {k: _replace(v, prompt, replaced) for k, v in value.items()}
    return value


def substitute_prompt(message: Any, prompt: str) -> Any:
    """Return a deep copy of ``message`` with every ``{{prompt}}`` replaced by ``prompt``.

    Raises ``ValueError`` if the placeholder is not present anywhere.
    """
    replaced = [False]
    result = _replace(copy.deepcopy(message), prompt, replaced)
    if not replaced[0]:
        raise ValueError(
            f"Template message contains no '{PROMPT_PLACEHOLDER}' placeholder to inject the prompt into"
        )
    return result


_TEMPLATE_CACHE: dict[str, "MappingTemplate[Any]"] = {}


def schema_cache_key(input_type: type) -> str:
    """Stable cache key: qualified class name + hash of its JSON schema."""
    schema = json.dumps(
        input_type.model_json_schema(),  # type: ignore[attr-defined]
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(schema.encode("utf-8")).hexdigest()[:16]
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
        key = schema_cache_key(input_type)
        cached = _TEMPLATE_CACHE.get(key)
        if cached is not None:
            return cached

        schema = json.dumps(input_type.model_json_schema(), indent=2, default=str)  # type: ignore[attr-defined]
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
