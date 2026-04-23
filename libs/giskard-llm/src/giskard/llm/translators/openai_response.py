import logging
from typing import TYPE_CHECKING, Any

from giskard.llm.types import (
    ResponseInputItem,
    ResponseOutputFunctionCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseResult,
    ToolDef,
    Usage,
)

from ..utils import deserialize_arguments, serialize_arguments

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
    def _input_to_openai(input: ResponseInputItem) -> "ResponseInputItemParam":
        if "type" not in input or input["type"] == "message":
            return input  # pyright: ignore[reportReturnType]

        if input["type"] == "function_call_output":
            return {**input}

        if input["type"] == "function_call":
            return {
                **input,
                "arguments": serialize_arguments(input["arguments"]),
            }

    @staticmethod
    def _inputs_to_openai(
        input: str | list[ResponseInputItem],
    ) -> "str | ResponseInputParam":
        if isinstance(input, list):
            return [OpenAIResponseTranslator._input_to_openai(item) for item in input]

        return input

    @staticmethod
    def to_openai(
        model: str,
        input: str | list[ResponseInputItem],
        *,
        instructions: str | None = None,
        previous_id: str | None = None,
        tools: list[ToolDef] | None = None,
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
                {"type": "function", **t["function"], "strict": None} for t in tools
            ]
        if params.get("temperature") is not None:
            response_params["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            response_params["max_output_tokens"] = params["max_tokens"]

        return response_params

    @staticmethod
    def from_openai(raw: "Response") -> ResponseResult:
        outputs: list[ResponseOutputItem] = []
        for item in raw.output:
            if item.type == "message":
                contents = []
                for content_block in item.content:
                    if content_block.type == "output_text":
                        contents.append(ResponseOutputText(text=content_block.text))
                    elif content_block.type == "refusal":
                        contents.append(
                            ResponseOutputRefusal(refusal=content_block.refusal)
                        )
                    else:
                        raise ValueError(
                            f"Unsupported message content block type: {content_block.type!r}"
                        )
                outputs.append(ResponseOutputMessage(content=contents, role=item.role))
            elif item.type == "function_call":
                outputs.append(
                    ResponseOutputFunctionCall(
                        call_id=item.call_id,
                        name=item.name,
                        arguments=deserialize_arguments(item.arguments),
                    )
                )
            else:
                raise ValueError(f"Unsupported item type: {item.type}")

        usage = None
        if raw.usage:
            usage = Usage(
                prompt_tokens=raw.usage.input_tokens,
                completion_tokens=raw.usage.output_tokens,
                total_tokens=raw.usage.total_tokens,
            )

        return ResponseResult(
            id=raw.id,
            outputs=outputs,
            model=getattr(raw, "model", None),
            usage=usage,
        )
