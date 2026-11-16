import logging
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from giskard.llm.types import (
    NormalizedReasoningEffort,
    ResponseInputItem,
    ResponseReasoningItem,
    ResponseResult,
    ToolDef,
)
from giskard.llm.types._base import _BaseModel
from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    field_validator,
    model_validator,
)

from ..errors import BadRequestError
from ..types._serialization import close_object_schemas
from ..utils import sanitize_schema_name
from ._unsupported import keep_supported

if TYPE_CHECKING:
    from openai.types.responses.response import Response
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsNonStreaming,
    )
    from openai.types.responses.tool_param import ToolParam

KNOWN_RESPONSE_PARAMS = frozenset(
    {"temperature", "max_tokens", "response_format", "reasoning_effort", "reasoning"}
)
_SUPPORTED_OUTPUT_TYPES = frozenset({"message", "function_call", "reasoning"})

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
            schema = close_object_schemas(deepcopy(schema))
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
            name = response_format["name"]
            schema = close_object_schemas(
                deepcopy(cast(dict[str, Any], response_format["schema"]))
            )
            return {
                "format": {
                    **response_format,
                    "schema": schema,
                }
            }
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
    reasoning: dict[str, Any] | None = None
    reasoning_effort: NormalizedReasoningEffort | None = Field(
        default=None, exclude=True
    )

    @field_validator("input", mode="after")
    @classmethod
    def _drop_foreign_reasoning(
        cls, value: str | Sequence[ResponseInputItem]
    ) -> str | Sequence[ResponseInputItem]:
        """Drop reasoning items another provider produced (tagged via ``format``).

        OpenAI cannot replay them: their ``id`` is not an ``rs_`` id (and can exceed
        64 chars) and ``encrypted_content`` is verified server-side. Untagged items
        are OpenAI-native and kept.
        """
        if isinstance(value, str):
            return value
        kept = [
            item
            for item in value
            if not (isinstance(item, ResponseReasoningItem) and item.format is not None)
        ]
        if len(kept) != len(value):
            logger.debug(
                "%s provider: dropped %d reasoning item(s) from another provider",
                PROVIDER,
                len(value) - len(kept),
            )
        return kept

    @model_validator(mode="before")
    @classmethod
    def _coerce_response_format(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        v = v.copy()
        response_format = v.pop("response_format", None)
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            schema = close_object_schemas(response_format.model_json_schema())
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

    @model_validator(mode="after")
    def _merge_reasoning(self) -> "OpenAIResponseParams":
        if self.reasoning_effort is None:
            return self
        if self.reasoning is None:
            self.reasoning = {"effort": self.reasoning_effort}
            return self
        if self.reasoning.get("effort") is not None:
            raise BadRequestError(
                400,
                "Cannot specify both reasoning_effort and reasoning['effort']",
                PROVIDER,
            )
        self.reasoning = {**self.reasoning, "effort": self.reasoning_effort}
        return self


class OpenAIResponseTranslator:
    @staticmethod
    def to_openai(
        model: str,
        input: str | Sequence[ResponseInputItem],
        *,
        instructions: str | None = None,
        previous_id: str | None = None,
        tools: Sequence[ToolDef] | None = None,
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
    def from_openai(
        raw: "Response",
        *,
        ignore_unsupported_content: bool = False,
        provider: str = PROVIDER,
    ) -> ResponseResult:
        data = raw.model_dump()
        data["output"] = keep_supported(
            data.get("output") or [],
            _SUPPORTED_OUTPUT_TYPES,
            provider,
            ignore_unsupported_content=ignore_unsupported_content,
        )
        if isinstance(usage := data.get("usage"), dict):
            details = usage.get("output_tokens_details")
            if (
                isinstance(details, dict)
                and (reasoning_tokens := details.get("reasoning_tokens")) is not None
            ):
                usage["reasoning_tokens"] = reasoning_tokens
        return ResponseResult.model_validate(data)
