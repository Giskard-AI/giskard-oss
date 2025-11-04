from __future__ import annotations

from typing import Any, Callable, TypeVar, get_origin

from pydantic import BaseModel, GetCoreSchemaHandler, computed_field
from pydantic_core import core_schema

DISCRIMINATOR = "kind"

T = TypeVar("T", bound=type)


class _Registry:
    def __init__(self):
        self.subclasses: dict[type, dict[str, type]] = {}
        self.reverse_subclasses: dict[type, type] = {}
        self.kinds: dict[type, str] = {}

    def register_base(self, base_cls: type):
        if not issubclass(base_cls, Discriminated):
            raise ValueError(f"Class {base_cls} is not a subclass of Discriminated")

        if base_cls in self.subclasses:
            raise ValueError(f"Class {base_cls} is already registered")

        self.subclasses[base_cls] = {}
        self.reverse_subclasses[base_cls] = base_cls

    def register_subclass(self, target_cls: type, base_cls: type, kind: str):
        if not issubclass(target_cls, base_cls):
            raise ValueError(f"Class {target_cls} is not a subclass of {base_cls}")

        if base_cls not in self.subclasses:
            raise ValueError(f"Class {base_cls} is not registered")

        if kind in self.subclasses[base_cls]:
            raise ValueError(f"Kind {kind} is already registered for {base_cls}")

        self.subclasses[base_cls][kind] = target_cls
        self.reverse_subclasses[target_cls] = base_cls
        self.kinds[target_cls] = kind

    def is_base_cls(self, cls: type) -> bool:
        if cls in self.subclasses:
            return True

        origin = get_origin(cls)
        if origin and origin in self.subclasses:
            return True

        meta = getattr(cls, "__pydantic_generic_metadata__", None)
        if isinstance(meta, dict) and meta.get("origin"):
            return meta["origin"] in self.subclasses

        return False

    def get_target_cls(self, cls: type, obj: Any) -> type:
        kind = (
            obj.get("kind", None)
            if isinstance(obj, dict)
            else getattr(obj, "kind", None)
        )

        if kind is None:
            raise ValueError(f"Kind is not provided for {cls}")

        # Resolve the base class for generic types
        base_cls = cls
        if cls not in self.subclasses:
            origin = get_origin(cls)
            if origin and origin in self.subclasses:
                base_cls = origin
            else:
                meta = getattr(cls, "__pydantic_generic_metadata__", None)
                if isinstance(meta, dict) and meta.get("origin") in self.subclasses:
                    base_cls = meta["origin"]

        if kind not in self.subclasses[base_cls]:
            raise ValueError(f"Kind {kind} is not registered for {base_cls}")

        return self.subclasses[base_cls][kind]

    def get_kind(self, cls: type) -> str | None:
        return self.kinds.get(cls, None)


_REGISTRY = _Registry()


class Discriminated(BaseModel):
    """Base class for discriminated union types.

    Provides a polymorphic type system with automatic serialization using
    a 'kind' discriminator field. Subclasses are registered and resolved
    automatically during validation.

    Examples
    --------
    >>> @discriminated_base
    ... class Animal(Discriminated):
    ...     pass
    >>>
    >>> @Animal.register("dog")
    ... class Dog(Animal):
    ...     breed: str
    >>>
    >>> dog = Dog(breed="Labrador")
    >>> dog.kind
    'dog'
    >>> serialized = dog.model_dump()
    >>> Animal.model_validate(serialized)  # Returns Dog instance
    """

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
        if cls in _REGISTRY.kinds:
            return _REGISTRY.kinds[cls]

        # Try to resolve the base class for generic types
        origin = get_origin(cls)
        if origin and origin in _REGISTRY.kinds:
            return _REGISTRY.kinds[origin]

        # Try pydantic generic metadata
        meta = getattr(cls, "__pydantic_generic_metadata__", None)
        if isinstance(meta, dict) and meta.get("origin") in _REGISTRY.kinds:
            return _REGISTRY.kinds[meta["origin"]]

        return None

    @classmethod
    def register(cls, kind: str) -> Callable[[T], T]:
        """Decorator to register a subclass with a kind identifier.

        Parameters
        ----------
        kind : str
            Unique identifier for this subclass.

        Returns
        -------
        Callable[[T], T]
            Decorator that registers the class.

        Examples
        --------
        >>> @Check.register("custom")
        ... class CustomCheck(Check):
        ...     pass
        """

        def decorator(target_cls: T) -> T:
            _REGISTRY.register_subclass(target_cls, cls, kind)
            return target_cls

        return decorator

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ):
        if _REGISTRY.is_base_cls(cls):
            return _REGISTRY.get_target_cls(cls, obj).model_validate(
                obj,
                strict=strict,
                from_attributes=from_attributes,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )

        return super().model_validate(
            obj,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        if not _REGISTRY.is_base_cls(cls):
            return handler(source)

        def validate_discriminated(value: Any) -> Any:
            if isinstance(value, Discriminated):
                return value

            return _REGISTRY.get_target_cls(cls, value).model_validate(value)

        return core_schema.no_info_plain_validator_function(validate_discriminated)


def discriminated_base(cls: T) -> T:
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
