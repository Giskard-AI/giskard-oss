import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from giskard.llm.types import (
    ResponseInputItem,
    ResponseResult,
    ToolDef,
)
from giskard.llm.types._base import _BaseModel
from pydantic import Field, SerializationInfo

if TYPE_CHECKING:
    from openai.types.responses.response import Response
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsNonStreaming,
    )
    from openai.types.responses.tool_param import ToolParam

KNOWN_RESPONSE_PARAMS = frozenset({"temperature", "max_tokens"})

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


class OpenAIResponseParams(_BaseModel):
    model: str
    input: str | Sequence[ResponseInputItem]
    instructions: str | None = None
    previous_response_id: str | None = None
    tools: Sequence[ToolDef] | None
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, validation_alias="max_tokens")


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
    def from_openai(raw: "Response") -> ResponseResult:
        return ResponseResult.model_validate(raw.model_dump())
