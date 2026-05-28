import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Required, TypedDict, cast

from giskard.llm.types._base import _BaseModel
from pydantic import BaseModel, SerializationInfo, field_serializer, model_validator

from ..types import (
    ResponseEasyInputMessage,
    ResponseFunctionCallOutput,
    ResponseFunctionToolCall,
    ResponseInputItem,
    ResponseInputText,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseResult,
    ToolDef,
    Usage,
)
from ..utils import deserialize_arguments

if TYPE_CHECKING:
    from google.genai._interactions.types import (
        GenerationConfigParam,
        Interaction,
        ModelOutputStepParam,
        StepParam,
        TextContentParam,
        ToolParam,
        UserInputStepParam,
        interaction_create_params,
    )
    from httpx import Timeout as httpxTimeout

    class InteractionCreateParams(TypedDict, total=False):
        input: Required[interaction_create_params.Input]
        model: Required[str]
        previous_interaction_id: str
        system_instruction: str
        timeout: float | httpxTimeout
        tools: Iterable[ToolParam]
        generation_config: GenerationConfigParam
        response_format: object
        response_mime_type: str
else:
    # Skip validation for httpxTimeout
    httpxTimeout = Any

_PROVIDER = "google/response"
PROVIDER = "google"
KNOWN_RESPONSE_PARAMS = frozenset({"temperature", "timeout", "response_format"})


logger = logging.getLogger(__name__)


@ToolDef.register_serializer(_PROVIDER)
def serialize_tool_def(tool: ToolDef, _info: SerializationInfo) -> "ToolParam":
    return {
        "type": "function",
        "name": tool.function.name,
        "description": tool.function.description or "No description provided",
        "parameters": tool.function.parameters or {},
    }


def _text_content(text: str) -> "TextContentParam":
    return {"type": "text", "text": text}


def _message_step(
    role: Literal["user", "assistant"],
    content: Iterable["TextContentParam"],
) -> "UserInputStepParam | ModelOutputStepParam":
    if role == "user":
        return {"content": content, "type": "user_input"}
    return {"content": content, "type": "model_output"}


@ResponseInputText.register_serializer(_PROVIDER)
def serialize_input_text(
    model: ResponseInputText, _info: SerializationInfo
) -> "TextContentParam":
    return _text_content(model.text)


@ResponseOutputText.register_serializer(_PROVIDER)
def serialize_output_text(
    model: ResponseOutputText, _info: SerializationInfo
) -> "TextContentParam":
    return _text_content(model.text)


@ResponseOutputRefusal.register_serializer(_PROVIDER)
def serialize_output_refusal(
    model: ResponseOutputRefusal, _info: SerializationInfo
) -> "TextContentParam":
    return _text_content(model.refusal)


@ResponseEasyInputMessage.register_serializer(_PROVIDER)
def serialize_easy_input_message(
    model: ResponseEasyInputMessage, info: SerializationInfo
) -> "StepParam | None":
    if model.role == "developer" or model.role == "system":
        return None

    if isinstance(model.content, str):
        return _message_step(model.role, [_text_content(model.content)])

    content = [
        cast("TextContentParam", cast(object, item.model_dump(context=info.context)))
        for item in model.content
    ]
    return _message_step(model.role, content)


@ResponseOutputMessage.register_serializer(_PROVIDER)
def serialize_output_message(
    model: ResponseOutputMessage, info: SerializationInfo
) -> "StepParam":
    if isinstance(model.content, str):
        return _message_step(model.role, [_text_content(model.content)])

    content = [
        cast("TextContentParam", cast(object, item.model_dump(context=info.context)))
        for item in model.content
    ]
    return _message_step(model.role, content)


@ResponseFunctionToolCall.register_serializer(_PROVIDER)
def serialize_output_function_call(
    model: ResponseFunctionToolCall, _info: SerializationInfo
) -> "StepParam":
    return {
        "type": "function_call",
        "id": model.call_id,
        "name": model.name,
        "arguments": deserialize_arguments(model.arguments),
    }


@ResponseFunctionCallOutput.register_serializer(_PROVIDER)
def serialize_output_function_call_output(
    model: ResponseFunctionCallOutput, _info: SerializationInfo
) -> "StepParam":
    if model.name is None:
        # We cannot compute name from call_id alone since function calls are not guaranteed to be in the input.
        raise ValueError("name is required for function calls")

    return {
        "type": "function_result",
        "call_id": model.call_id,
        "name": model.name,
        "result": model.output,
    }


def _extract_system_instruction(input: str | Sequence[ResponseInputItem]) -> str | None:
    if isinstance(input, str):
        return None

    system_parts = [
        part.text
        for part in input
        if (part.type is None or part.type == "message")
        and part.role in ("system", "developer")
    ]
    system_parts = [part for part in system_parts if part is not None]

    return "\n".join(system_parts) if system_parts else None


class GoogleResponseGenerationConfigParam(_BaseModel):
    temperature: float | None = None


class GoogleResponseParams(_BaseModel):
    model: str
    input: str | Sequence[ResponseInputItem]
    system_instruction: str | None = None
    previous_interaction_id: str | None = None
    tools: Sequence[ToolDef] | None
    generation_config: GoogleResponseGenerationConfigParam | None = None
    timeout: float | httpxTimeout | None = None
    response_mime_type: Literal["application/json"] | None = None
    response_format: type[BaseModel] | dict[str, Any] | None = None

    @field_serializer("input")
    def serialize_input(
        self, value: str | Sequence[ResponseInputItem], info: SerializationInfo
    ) -> Any:
        if isinstance(value, str):
            return value

        inputs = [item.model_dump(context=info.context) for item in value]
        return [item for item in inputs if item is not None]

    @model_validator(mode="before")
    @classmethod
    def _coerce_dict(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v

        v = v.copy()

        # Extract system instruction from input and merge with instructions
        instructions_parts = [
            v.get("system_instruction"),
            _extract_system_instruction(v["input"]),
        ]
        instructions_parts = [part for part in instructions_parts if part is not None]
        if instructions_parts:
            v["system_instruction"] = "\n".join(instructions_parts)

        # Move temperature to generation_config
        if "temperature" in v:
            v["generation_config"] = {"temperature": v.pop("temperature")}

        # Setup response_format for JSON output
        if (
            "response_format" in v
            and isinstance(v["response_format"], type)
            and issubclass(v["response_format"], BaseModel)
        ):
            v["response_mime_type"] = "application/json"

        return v


class GoogleResponseTranslator:
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

        google_params = GoogleResponseParams.model_validate(
            {
                "model": model,
                "input": input,
                "system_instruction": instructions,
                "previous_interaction_id": previous_id,
                "tools": tools,
                **params,
            }
        )

        return cast(
            "InteractionCreateParams",
            cast(
                object,
                google_params.model_dump(context={"provider": _PROVIDER}),
            ),
        )

    @staticmethod
    def from_google(raw: "Interaction", model: str) -> ResponseResult:
        outputs: list[ResponseOutputItem] = []
        for item in raw.steps or []:
            if item.type == "model_output":
                for content in item.content or []:
                    if content.type == "text":
                        outputs.append(
                            ResponseOutputMessage(
                                content=[ResponseOutputText(text=content.text)],
                                role="assistant",
                            )
                        )
            elif item.type == "function_call":
                outputs.append(
                    ResponseFunctionToolCall(
                        call_id=item.id,
                        name=item.name,
                        arguments=item.arguments,
                    )
                )

        usage = None
        if raw.usage:
            usage = Usage(
                input_tokens=raw.usage.total_input_tokens or 0,
                output_tokens=raw.usage.total_output_tokens or 0,
                total_tokens=raw.usage.total_tokens or 0,
            )

        return ResponseResult(
            id=raw.id,
            outputs=outputs,
            model=model,
            usage=usage,
        )
