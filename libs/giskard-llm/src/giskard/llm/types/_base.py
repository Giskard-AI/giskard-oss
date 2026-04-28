import json
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    FieldSerializationInfo,
    PlainSerializer,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from ._serialization import get_serializer, register_serializer


class _BaseModel(BaseModel):
    """Shared base for all giskard-llm response models. Defaults model_dump to exclude None fields."""

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    @model_serializer(mode="wrap")
    def serialize(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> Any:
        provider = (
            info.context.get("provider") if isinstance(info.context, dict) else None
        )
        custom_serializer = (
            get_serializer(self.__class__, provider) if provider else None
        )

        if custom_serializer:
            return custom_serializer(self, info)

        return handler(self)

    @classmethod
    def register_serializer(cls, provider: str):
        return register_serializer(cls, provider)


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


_JSON_PROVIDERS = frozenset({"openai/chat", "openai/response"})


def _serialize_json(value: Any, info: FieldSerializationInfo) -> Any:
    provider = info.context.get("provider") if isinstance(info.context, dict) else None
    if provider in _JSON_PROVIDERS:
        return json.dumps(value)

    return value


ArgumentDict = Annotated[
    dict[str, object], BeforeValidator(_coerce_json), PlainSerializer(_serialize_json)
]
