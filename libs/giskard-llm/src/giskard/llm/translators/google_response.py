import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Required, TypedDict

from pydantic import BaseModel

from ..types import (
    ResponseInputItem,
    ResponseOutputFunctionCall,
    ResponseOutputText,
    ResponseResult,
    ToolDef,
    Usage,
)

if TYPE_CHECKING:
    import httpx
    from google.genai._interactions.types import (
        FunctionCallContentParam,
        FunctionResultContentParam,
        GenerationConfigParam,
        Interaction,
        ToolParam,
        interaction_create_params,
    )

    class InteractionCreateParams(TypedDict, total=False):
        input: Required[interaction_create_params.Input]
        model: Required[str]
        previous_interaction_id: str
        system_instruction: str
        timeout: float | httpx.Timeout
        tools: Iterable[ToolParam]
        generation_config: GenerationConfigParam
        response_format: object
        response_mime_type: str


PROVIDER = "google"
KNOWN_RESPONSE_PARAMS = frozenset({"temperature", "timeout", "response_format"})

logger = logging.getLogger(__name__)


class GoogleResponseTranslator:
    @staticmethod
    def _input_to_google(
        input: ResponseInputItem,
    ) -> "FunctionResultContentParam | FunctionCallContentParam":
        if input["type"] == "function_call_output":
            name = input.get("name")
            if name is None:
                raise ValueError("function_call_output: name is required")

            return {
                "type": "function_result",
                "call_id": input["call_id"],
                "name": name,
                "result": input["output"],
            }

        if input["type"] == "function_call":
            id = input.get("id")
            if id is None:
                raise ValueError("function_call: id is required")
            return {
                "type": "function_call",
                "id": id,
                "name": input.get("name", ""),
                "arguments": {},
            }

    @staticmethod
    def _inputs_to_google(
        input: str | list[ResponseInputItem],
    ) -> "interaction_create_params.Input":
        if isinstance(input, list):
            return [GoogleResponseTranslator._input_to_google(item) for item in input]

        return input

    @staticmethod
    def to_google(
        model: str,
        input: str | list[ResponseInputItem],
        *,
        instructions: str | None = None,
        previous_id: str | None = None,
        tools: list[ToolDef] | None = None,
        **params: Any,
    ) -> "InteractionCreateParams":
        unknown = set(params) - KNOWN_RESPONSE_PARAMS
        if unknown:
            logger.warning(
                "%s provider: ignoring unknown response params: %s",
                PROVIDER,
                sorted(unknown),
            )

        interaction_create_params: "InteractionCreateParams" = {
            "model": model,
            "input": GoogleResponseTranslator._inputs_to_google(input),
        }

        if instructions is not None:
            interaction_create_params["system_instruction"] = instructions
        if previous_id is not None:
            interaction_create_params["previous_interaction_id"] = previous_id
        if tools is not None:
            interaction_create_params["tools"] = [
                {"type": "function", **t["function"]} for t in tools
            ]
        if params.get("temperature") is not None:
            interaction_create_params["generation_config"] = (
                interaction_create_params.get("generation_config", {})
            )
            interaction_create_params["generation_config"]["temperature"] = params[
                "temperature"
            ]
        if params.get("timeout") is not None:
            interaction_create_params["timeout"] = params["timeout"]

        response_format = params.get("response_format")
        if (
            response_format is not None
            and isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
        ):
            interaction_create_params["response_mime_type"] = "application/json"
            interaction_create_params["response_format"] = response_format

        return interaction_create_params

    @staticmethod
    def from_google(raw: "Interaction", model: str) -> ResponseResult:
        outputs: list[ResponseOutputText | ResponseOutputFunctionCall] = []
        for item in getattr(raw, "outputs", []):
            item_type = getattr(item, "type", None)
            if item_type == "text":
                outputs.append(ResponseOutputText(text=item.text))
            elif item_type == "function_call":
                args = getattr(item, "arguments", {})
                # Google returns "id" on function_call outputs, not "call_id"
                call_id = getattr(item, "id", None) or getattr(item, "call_id", None)
                outputs.append(
                    ResponseOutputFunctionCall(
                        call_id=call_id,
                        name=item.name,
                        arguments=args if isinstance(args, dict) else {},
                    )
                )

        usage = None
        usage_meta = getattr(raw, "usage", None)
        if usage_meta:
            outputs_tokens = getattr(usage_meta, "output_tokens", 0) or 0
            input_tokens = getattr(usage_meta, "input_tokens", 0) or 0
            usage = Usage(
                prompt_tokens=input_tokens,
                completion_tokens=outputs_tokens,
                total_tokens=input_tokens + outputs_tokens,
            )

        return ResponseResult(
            id=raw.id,
            outputs=outputs,
            model=model,
            usage=usage,
        )
