import base64
import binascii
import json
from typing import Annotated, Any, ClassVar

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

    # Fields kept on the model but never sent to a provider API.
    _wire_excluded: ClassVar[frozenset[str]] = frozenset()

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

        data = handler(self)
        if provider and self._wire_excluded and isinstance(data, dict):
            for field in self._wire_excluded:
                data.pop(field, None)
        return data

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


def _coerce_b64_bytes(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return base64.b64decode(value.encode("ascii"), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid base64-encoded bytes") from exc
    return value


def _serialize_b64_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


Base64Bytes = Annotated[
    bytes,
    BeforeValidator(_coerce_b64_bytes),
    PlainSerializer(_serialize_b64_bytes, return_type=str),
]
