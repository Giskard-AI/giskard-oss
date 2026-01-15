import sys
from collections import defaultdict
from typing import Any, Callable, Generic, TypeVar

from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    computed_field,
)
from pydantic_core import core_schema

T = TypeVar("T")

CURRENT_MODULE_NAME = sys.modules[__name__].__name__


class _Registry(Generic[T]):
    def __init__(self):
        self._subclasses: dict[type, dict[str, type]] = {}
        self._kinds: dict[type[T], str] = {}
        self._reverse_kinds: dict[type[T], dict[str, type[T]]] = defaultdict(dict)

    def register_base(self, base_cls: type[T]):
        if not issubclass(base_cls, Discriminated):
            raise ValueError(f"Class {base_cls} is not a subclass of Discriminated")

        if base_cls in self._subclasses:
            raise ValueError(f"Class {base_cls} is already registered")

        self._subclasses[base_cls] = {}

    def _get_base_cls(self, cls: type[T]) -> type[T] | None:
        if cls in self._subclasses:
            return cls

        for base in cls.__bases__:
            subclass = self._get_base_cls(base)
            if subclass is not None:
                return subclass

        return None

    def register_subclass(self, base_cls: type[T], subclass: type[T], kind: str):
        if not issubclass(subclass, base_cls):
            raise ValueError(f"Class {subclass} is not a subclass of {base_cls}")

        actual_base_cls = self._get_base_cls(base_cls)

        if actual_base_cls is None:
            raise ValueError(
                f"Class {base_cls} is not registered with @discriminated_base"
            )

        if kind in self._subclasses[actual_base_cls]:
            raise ValueError(f"Kind {kind} is already registered for {base_cls}")

        self._subclasses[actual_base_cls][kind] = subclass
        self._kinds[subclass] = kind
        self._reverse_kinds[actual_base_cls][kind] = subclass


_REGISTRY = _Registry()


class Discriminated(BaseModel):
    @computed_field
    def kind(self) -> str | None:
        """The discriminator field identifying the concrete type.

        Returns
        -------
        str | None
            The kind string registered for this class, or None if unregistered.
        """
        cls = self.__class__

        # Check if the class is directly registered
        if cls in _REGISTRY._kinds:
            return _REGISTRY._kinds[cls]

        return None

    @classmethod
    def register(cls, kind: str) -> Callable[[type[T]], type[T]]:
        def decorator(subclass: type[T]) -> type[T]:
            _REGISTRY.register_subclass(cls, subclass, kind)
            return subclass

        return decorator

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        metadata = getattr(cls, "__pydantic_generic_metadata__", {})
        origin = metadata.get("origin") or cls

        if not any(
            base.__name__ == "Discriminated" and base.__module__ == CURRENT_MODULE_NAME
            for base in origin.__bases__
        ):
            return handler(source)

        def validate_discriminated(value: Any) -> Any:
            if isinstance(value, Discriminated):
                return value
            elif not isinstance(value, dict):
                raise ValueError(f"Value {value} is not a dictionary")

            if "kind" not in value:
                raise ValueError(f"Kind is not provided for class {origin}")

            kind = value["kind"]
            if not isinstance(kind, str):
                raise ValueError(f"Kind is expected to be a string, got {type(kind)}")

            if kind not in _REGISTRY._reverse_kinds[origin]:
                raise ValueError(f"Kind {kind} is not registered for class {origin}")

            return _REGISTRY._reverse_kinds[origin][kind].model_validate(value)

        return core_schema.no_info_plain_validator_function(validate_discriminated)


def discriminated_base(cls: type[T]) -> type[T]:
    """Mark a class as the base of a discriminated union.

    Use this decorator on base classes that will have multiple concrete
    implementations registered with different 'kind' values.

    Parameters
    ----------
    cls : T
        The base class to register.

    Returns
    -------
    T
        The same class, now registered as a discriminated base.

    Examples
    --------
    >>> @discriminated_base
    ... class Check(Discriminated):
    ...     async def run(self, interaction): ...
    """
    _REGISTRY.register_base(cls)
    return cls
