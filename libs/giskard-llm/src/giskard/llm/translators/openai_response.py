import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from giskard.llm.types import (
    ResponseInputItem,
    ResponseResult,
    ToolDef,
)
from giskard.llm.types._base import _BaseModel
from pydantic import BaseModel, Field, SerializationInfo, model_validator

from ..errors import BadRequestError
from ..structured_output import (
    DEFAULT_SCHEMA_MUTATION,
    SchemaMutationMode,
    normalize_pydantic_json_schema,
    pop_schema_mutation,
)
from ..utils import sanitize_schema_name

if TYPE_CHECKING:
    from openai.types.responses.response import Response
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsNonStreaming,
    )
    from openai.types.responses.tool_param import ToolParam

KNOWN_RESPONSE_PARAMS = frozenset({"temperature", "max_tokens", "response_format"})

logger = logging.getLogger(__name__)
PROVIDER = "openai"
_PROVIDER = "openai/response"


@ToolDef.register_serializer(_PROVIDER)
def tool_def_to_openai(tool: ToolDef, _info: SerializationInfo) -> "ToolParam":
    return {
        "type": "function",
        "name": tool.function.name,
        "description": tool.function.description,
        "parameters": tool.function.parameters,
        "strict": None,
    }


def _text_config_from_response_format_dict(
    response_format: dict[str, Any],
) -> dict[str, Any]:
    """Map OpenAI-shaped ``response_format`` dict to Responses ``text`` config."""
    rf_type = response_format.get("type")
    if rf_type == "json_schema":
        if "json_schema" in response_format:
            inner = response_format["json_schema"]
            if not isinstance(inner, dict):
                raise BadRequestError(
                    400,
                    "response_format json_schema must be an object",
                    PROVIDER,
                )
            name = inner.get("name")
            schema = inner.get("schema")
            if not isinstance(name, str) or not isinstance(schema, dict):
                raise BadRequestError(
                    400,
                    "response_format json_schema must include string name and object schema",
                    PROVIDER,
                )
            fmt: dict[str, Any] = {
                "type": "json_schema",
                "name": name,
                "schema": schema,
            }
            if "strict" in inner:
                fmt["strict"] = inner["strict"]
            return {"format": fmt}
        if isinstance(response_format.get("name"), str) and isinstance(
            response_format.get("schema"), dict
        ):
            return {"format": response_format}
        raise BadRequestError(
            400,
            "response_format type json_schema must use Chat Completions "
            "(json_schema.name/schema) or Responses (name/schema) shape",
            PROVIDER,
        )
    if rf_type in ("json_object", "text"):
        return {"format": response_format}
    raise BadRequestError(
        400,
        f"Unsupported response_format type {rf_type!r} for OpenAI Responses API",
        PROVIDER,
    )


class OpenAIResponseParams(_BaseModel):
    model: str
    input: str | Sequence[ResponseInputItem]
    instructions: str | None = None
    previous_response_id: str | None = None
    tools: Sequence[ToolDef] | None
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, validation_alias="max_tokens")
    text: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_response_format(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        v = v.copy()
        mode = pop_schema_mutation(v)
        response_format = v.pop("response_format", None)
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            schema = normalize_pydantic_json_schema(
                response_format,
                profile="openai",
                provider=PROVIDER,
                mode=mode,
            )
            v["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": sanitize_schema_name(response_format.__name__),
                    "schema": schema,
                }
            }
        elif isinstance(response_format, dict):
            v["text"] = _text_config_from_response_format_dict(response_format)
        elif response_format is not None:
            raise BadRequestError(
                400,
                "response_format must be a Pydantic model class or a dict",
                PROVIDER,
            )
        return v


class OpenAIResponseTranslator:
    @staticmethod
    def to_openai(
        model: str,
        input: str | Sequence[ResponseInputItem],
        *,
        instructions: str | None = None,
        previous_id: str | None = None,
        tools: Sequence[ToolDef] | None = None,
        schema_mutation: SchemaMutationMode = DEFAULT_SCHEMA_MUTATION,
        **params: Any,
    ) -> "ResponseCreateParamsNonStreaming":
        unknown = set(params) - KNOWN_RESPONSE_PARAMS
        if unknown:
            logger.warning(
                "%s provider: ignoring unknown response params: %s",
                PROVIDER,
                sorted(unknown),
            )

        response_params = OpenAIResponseParams.model_validate(
            {
                "model": model,
                "input": input,
                "instructions": instructions,
                "previous_response_id": previous_id,
                "tools": tools,
                "schema_mutation": schema_mutation,
                **params,
            }
        )

        return cast(
            "ResponseCreateParamsNonStreaming",
            cast(
                object,
                response_params.model_dump(context={"provider": _PROVIDER}),
            ),
        )

    @staticmethod
    def from_openai(raw: "Response") -> ResponseResult:
        return ResponseResult.model_validate(raw.model_dump())
