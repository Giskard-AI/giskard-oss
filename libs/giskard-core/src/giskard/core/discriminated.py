import sys
from collections.abc import Sequence
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

    def register_subclass(
        self,
        base_cls: type[T],
        subclass: type[T],
        kind: str,
        aliases: Sequence[str] = (),
    ):
        if not issubclass(subclass, base_cls):
            raise ValueError(f"Class {subclass} is not a subclass of {base_cls}")

        actual_base_cls = self._get_base_cls(base_cls)

        if actual_base_cls is None:
            raise ValueError(
                f"Class {base_cls} is not registered with @discriminated_base"
            )

        registered = self._subclasses[actual_base_cls]
        kinds = (kind, *aliases)

        # Validate every kind before writing any, so a collision partway through
        # the aliases cannot leave the registry half-populated.
        for registered_kind in kinds:
            if registered_kind in registered:
                raise ValueError(
                    f"Kind {registered_kind} is already registered for {base_cls}"
                )

        # Only the canonical kind is mapped back from the class, so aliases are
        # accepted when deserialising but never used when serialising.
        self._kinds[subclass] = kind

        for registered_kind in kinds:
            registered[registered_kind] = subclass


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
    def register(
        cls, kind: str, *, aliases: Sequence[str] = ()
    ) -> Callable[[type[T]], type[T]]:
        """Register a subclass under a discriminator kind.

        Parameters
        ----------
        kind : str
            The canonical discriminator. This is the value written to the
            ``kind`` field when the subclass is serialised.
        aliases : Sequence[str]
            Additional discriminators accepted when deserialising, typically
            legacy names kept for backward compatibility. Payloads using an
            alias load as this subclass but are re-serialised under ``kind``.

        Examples
        --------
        >>> @Check.register("less_than", aliases=["lesser_than"])
        ... class LessThan(Check): ...
        """
        if isinstance(aliases, str):
            raise TypeError(
                f"aliases must be a sequence of strings, got the string {aliases!r}; "
                f"pass [{aliases!r}] instead."
            )
        for alias in aliases:
            if not isinstance(alias, str):
                raise TypeError(
                    f"aliases must be a sequence of strings, got {alias!r} "
                    f"of type {type(alias).__name__}."
                )

        def decorator(subclass: type[T]) -> type[T]:
            _REGISTRY.register_subclass(cls, subclass, kind, aliases)
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

            registered = _REGISTRY._subclasses.get(origin)
            if registered is None or kind not in registered:
                raise ValueError(f"Kind {kind} is not registered for class {origin}")

            return registered[kind].model_validate(value)

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
