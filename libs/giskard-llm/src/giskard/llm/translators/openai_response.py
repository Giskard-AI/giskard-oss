import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from giskard.llm.types import (
    ResponseInputItem,
    ResponseResult,
    ToolDef,
)

if TYPE_CHECKING:
    from openai.types.responses.response import Response
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsNonStreaming,
    )
    from openai.types.responses.response_input_param import (
        ResponseInputItemParam,
        ResponseInputParam,
    )

KNOWN_RESPONSE_PARAMS = frozenset({"temperature", "max_tokens"})

logger = logging.getLogger(__name__)
PROVIDER = "openai"


class OpenAIResponseTranslator:
    @staticmethod
    def _input_to_openai(
        input: ResponseInputItem,
    ) -> "ResponseInputItemParam":
        return cast(
            "ResponseInputItemParam",
            cast(object, input.model_dump(context={"json_arguments": True})),
        )

    @staticmethod
    def _inputs_to_openai(
        input: str | Sequence[ResponseInputItem],
    ) -> "str | ResponseInputParam":
        if isinstance(input, str):
            return input

        return [OpenAIResponseTranslator._input_to_openai(item) for item in input]

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

        response_params: "ResponseCreateParamsNonStreaming" = {
            "model": model,
            "input": OpenAIResponseTranslator._inputs_to_openai(input),
        }

        if instructions is not None:
            response_params["instructions"] = instructions
        if previous_id is not None:
            response_params["previous_response_id"] = previous_id
        if tools is not None:
            response_params["tools"] = [
                {
                    "type": "function",
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters,
                    "strict": None,
                }
                for t in tools
            ]
        if params.get("temperature") is not None:
            response_params["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            response_params["max_output_tokens"] = params["max_tokens"]

        return response_params

    @staticmethod
    def from_openai(raw: "Response") -> ResponseResult:
        return ResponseResult.model_validate(raw.model_dump())
