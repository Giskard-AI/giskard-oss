from __future__ import annotations

import os
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, computed_field, model_validator

from .registry import Registry

"""Generic interaction model.

An `Interaction` represents an input, an optional output (e.g., a model
response), and optional metadata captured during evaluation. Concrete
specializations can refine the input/output types as needed.
"""

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Interaction(BaseModel, Generic[InputT, OutputT]):
    """Container for a single interaction under test.

    Attributes
    ----------
    input:
        The input payload for the system under test.
    output:
        Optional output produced by the system.
    metadata:
        Optional free-form metadata associated with the interaction.
    """

    # Preferred usage: set a class-level KIND on subclasses, e.g. KIND = "chat".
    # This removes the need to pass `kind` to each instance and allows registry validation.
    KIND: ClassVar[str | None] = None

    input: InputT
    output: OutputT | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _populate_and_validate_kind(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        provided = data.get("kind")
        class_kind = getattr(cls, "KIND", None)
        if class_kind is not None:  # Allow None for base Interaction class
            if not (isinstance(class_kind, str) and class_kind):
                raise ValueError(f"KIND must be set for {cls.__name__}")
            if provided is not None and provided != class_kind:
                raise ValueError(
                    f"kind '{provided}' does not match class KIND '{class_kind}' for {cls.__name__}"
                )
        return data

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Skip the exact base class and any abstract intermediates
        if cls is Interaction or getattr(cls, "__abstractmethods__", None):
            return

        # Skip generic instantiations (like StructuredInteraction[str, str])
        # Only register the actual class definitions
        if (
            hasattr(cls, "__origin__")
            or getattr(cls, "__args__", None)
            or "[" in cls.__name__
        ):
            return

        class_kind = getattr(cls, "KIND", None)
        if isinstance(class_kind, str) and class_kind:
            _INTERACTION_REGISTRY.register(class_kind, cls)

    @computed_field(return_type=str | None)
    @property
    def kind(self) -> str | None:
        """Return the interaction type identifier from the class `KIND`.

        Marked as a computed field so it's included in serialization via
        `model_dump()` and JSON dumps, enabling generic deserialization.

        Returns None for the base Interaction class.
        """
        class_kind = getattr(self.__class__, "KIND", None)
        if class_kind is not None and not (isinstance(class_kind, str) and class_kind):
            raise ValueError(f"KIND must be set for {self.__class__.__name__}")
        return class_kind

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "Interaction[Any, Any]":
        """Instantiate a concrete `Interaction` from serialized data.

        Uses registry-based format with `kind` field for deserialization.
        """
        kind = data.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(
                "Serialized interaction must include a non-empty 'kind' field"
            )

        target_cls = _INTERACTION_REGISTRY.get_or_raise(kind)
        return target_cls.model_validate(data)

    def serialize(self) -> dict[str, Any]:
        """Serialize the interaction into a JSON-friendly dict.

        The output includes the computed `kind` field to enable registry-based
        deserialization.
        """
        return self.model_dump()


# Global registry for Interaction kinds
_ENFORCE_KIND_UNIQUENESS: bool = os.getenv(
    "GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS", "1"
).lower() in {"1", "true", "yes", "on"}

_INTERACTION_REGISTRY = Registry[Interaction[Any, Any]](
    name="interaction", enforce_uniqueness=_ENFORCE_KIND_UNIQUENESS
)


def list_registered_interaction_kinds() -> list[str]:
    """List all registered interaction kinds in alphabetical order.

    Returns
    -------
    list[str]
        Sorted list of all registered interaction kind identifiers
    """
    return _INTERACTION_REGISTRY.list_kinds()
