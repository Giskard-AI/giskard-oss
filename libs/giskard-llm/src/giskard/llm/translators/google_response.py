import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Required, TypedDict

from pydantic import BaseModel

from ..types import (
    ResponseInputItem,
    ResponseInputMessageContent,
    ResponseOutputFunctionCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputMessageContent,
    ResponseOutputText,
    ResponseResult,
    ToolDef,
    Usage,
)
from ..utils import deserialize_arguments

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
    from google.genai._interactions.types import (
        TextContentParam as GoogleTextContentParam,
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


def _flatten[T](items: Sequence[Sequence[T]]) -> list[T]:
    return [item for sublist in items for item in sublist]


class GoogleResponseTranslator:
    @staticmethod
    def _output_content_to_google(
        content: (ResponseInputMessageContent | ResponseOutputMessageContent),
    ) -> "GoogleTextContentParam":
        if content.type == "input_text":
            return {"type": "text", "text": content.text}
        if content.type == "output_text":
            return {"type": "text", "text": content.text}
        if content.type == "refusal":
            return {"type": "text", "text": content.refusal}
        # Runtime guard: inputs may not match TypedDict at runtime.
        raise ValueError(f"Unsupported message content type: {content.type!r}")  # pyright: ignore[reportUnreachable]

    @staticmethod
    def _content_to_google(
        content: str
        | Sequence[ResponseInputMessageContent | ResponseOutputMessageContent],
    ) -> "list[GoogleTextContentParam]":
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        return [
            GoogleResponseTranslator._output_content_to_google(item) for item in content
        ]

    @staticmethod
    def _input_to_google(
        input: ResponseInputItem,
    ) -> "Sequence[FunctionResultContentParam | FunctionCallContentParam | GoogleTextContentParam]":
        if input.type == "message":
            if input.role == "developer" or input.role == "system":
                return []  # Those messages are folded into the system instruction

            return GoogleResponseTranslator._content_to_google(input.content)

        if input.type == "function_call_output":
            name = input.name
            if name is None:
                raise ValueError("function_call_output: name is required")

            return [
                {
                    "type": "function_result",
                    "call_id": input.call_id,
                    "name": name,
                    "result": input.output,
                }
            ]

        if input.type == "function_call":
            return [
                {
                    "type": "function_call",
                    "id": input.call_id,
                    "name": input.name,
                    "arguments": deserialize_arguments(input.arguments),
                }
            ]

        raise ValueError(f"Unsupported input type: {input.type!r}")

    @staticmethod
    def _inputs_to_google(
        input: str | Sequence[ResponseInputItem],
    ) -> "interaction_create_params.Input":
        if isinstance(input, str):
            return input

        return _flatten(
            [GoogleResponseTranslator._input_to_google(item) for item in input]
        )

    @staticmethod
    def _extract_system_instruction(
        input: str | Sequence[ResponseInputItem],
    ) -> "str | None":
        if isinstance(input, str):
            return None

        system_parts = _flatten(
            [
                [
                    part["text"]
                    for part in GoogleResponseTranslator._content_to_google(
                        item.content
                    )
                ]
                for item in input
                if (item.type == "message") and item.role in ("system", "developer")
            ]
        )

        return "\n".join(system_parts) if system_parts else None

    @staticmethod
    def to_google(
        model: str,
        input: str | Sequence[ResponseInputItem],
        *,
        instructions: str | None = None,
        previous_id: str | None = None,
        tools: Sequence[ToolDef] | None = None,
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

        instructions_parts = [
            instructions,
            GoogleResponseTranslator._extract_system_instruction(input),
        ]
        instructions_parts = [part for part in instructions_parts if part is not None]

        if instructions_parts:
            interaction_create_params["system_instruction"] = "\n".join(
                instructions_parts
            )
        if previous_id is not None:
            interaction_create_params["previous_interaction_id"] = previous_id
        if tools is not None:
            interaction_create_params["tools"] = [
                {
                    "type": "function",
                    "name": t.function.name,
                    "description": t.function.description or "",
                    "parameters": t.function.parameters or {},
                }
                for t in tools
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
        outputs: list[ResponseOutputItem] = []
        for item in getattr(raw, "outputs", []):
            item_type = getattr(item, "type", None)
            if item_type == "text":
                outputs.append(
                    ResponseOutputMessage(
                        content=[ResponseOutputText(text=item.text)], role="assistant"
                    )
                )
            elif item_type == "function_call":
                raw_args = getattr(item, "arguments", None)
                if raw_args is None:
                    arguments: dict[str, Any] = {}
                elif isinstance(raw_args, (str, dict)):
                    arguments = deserialize_arguments(raw_args)
                else:
                    arguments = {}
                # Google returns "id" on function_call outputs, not "call_id"
                call_id = getattr(item, "id", None) or getattr(item, "call_id", None)
                outputs.append(
                    ResponseOutputFunctionCall(
                        call_id=call_id,
                        name=item.name,
                        arguments=arguments,
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
