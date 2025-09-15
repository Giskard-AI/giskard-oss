"""Registry infrastructure for checks and interactions.

This module provides the foundational registry system that replaces dynamic
imports with explicit registration patterns. All checks and interactions must
be registered before they can be serialized and deserialized.
"""

from __future__ import annotations

import warnings
from typing import Any, Generic, TypeVar

__all__ = ["Registry", "UnknownKindError", "DuplicateKindError"]

T = TypeVar("T")


class UnknownKindError(ValueError):
    """Raised when attempting to deserialize an unregistered kind.

    This error provides actionable guidance for developers encountering
    serialization issues with unregistered types.
    """

    def __init__(
        self, kind: str, registry_name: str, available_kinds: list[str]
    ) -> None:
        available_str = ", ".join(f"'{k}'" for k in sorted(available_kinds))
        super().__init__(
            f"Unknown {registry_name} kind '{kind}'. "
            f"Available kinds: {available_str}. "
            f"Make sure the class defining this kind is imported and properly registered."
        )
        self.kind = kind
        self.registry_name = registry_name
        self.available_kinds = available_kinds


class DuplicateKindError(ValueError):
    """Raised when attempting to register a kind that already exists.

    This error helps developers identify conflicts in kind registration
    during development and testing.
    """

    def __init__(self, kind: str, existing_class: type, new_class: type) -> None:
        super().__init__(
            f"Duplicate kind '{kind}' detected. "
            f"Already registered by {existing_class.__module__}.{existing_class.__name__}, "
            f"attempted to register {new_class.__module__}.{new_class.__name__}."
        )
        self.kind = kind
        self.existing_class = existing_class
        self.new_class = new_class


class Registry(Generic[T]):
    """Base registry for managing kinds and their associated classes.

    Provides registration, retrieval, and listing utilities with
    configurable duplicate handling and clear error messages.
    """

    def __init__(self, name: str, enforce_uniqueness: bool = True):
        """Initialize a new registry.

        Parameters
        ----------
        name : str
            Human-readable name for the registry (used in error messages)
        enforce_uniqueness : bool, default=True
            If True, raises DuplicateKindError on duplicate registrations.
            If False, emits a warning and overwrites the existing registration.
        """
        self._name = name
        self._enforce_uniqueness = enforce_uniqueness
        self._kinds: dict[str, type[T]] = {}

    def register(self, kind: str, cls: type[T]) -> None:
        """Register a class for the given kind.

        Parameters
        ----------
        kind : str
            The kind identifier to register
        cls : type[T]
            The class to associate with this kind

        Raises
        ------
        DuplicateKindError
            If enforce_uniqueness is True and the kind is already registered
            with a different class
        """
        existing = self._kinds.get(kind)
        if existing is not None and existing is not cls:
            if self._enforce_uniqueness:
                raise DuplicateKindError(kind, existing, cls)
            warnings.warn(
                f"Duplicate {self._name} kind '{kind}' detected for "
                f"{existing.__module__}.{existing.__name__} and "
                f"{cls.__module__}.{cls.__name__}; "
                "latest class will overwrite the registry entry.",
                RuntimeWarning,
                stacklevel=3,
            )
        self._kinds[kind] = cls

    def get(self, kind: str) -> type[T] | None:
        """Retrieve the class registered for the given kind.

        Parameters
        ----------
        kind : str
            The kind identifier to look up

        Returns
        -------
        type[T] | None
            The registered class, or None if not found
        """
        return self._kinds.get(kind)

    def get_or_raise(self, kind: str) -> type[T]:
        """Retrieve the class registered for the given kind, or raise an error.

        Parameters
        ----------
        kind : str
            The kind identifier to look up

        Returns
        -------
        type[T]
            The registered class

        Raises
        ------
        UnknownKindError
            If the kind is not registered
        """
        cls = self.get(kind)
        if cls is None:
            raise UnknownKindError(kind, self._name, self.list_kinds())
        return cls

    def list_kinds(self) -> list[str]:
        """List all registered kinds in alphabetical order.

        Returns
        -------
        list[str]
            Sorted list of all registered kind identifiers
        """
        return sorted(self._kinds.keys())

    def clear(self) -> None:
        """Clear all registrations (primarily for testing)."""
        self._kinds.clear()

    @property
    def name(self) -> str:
        """The human-readable name of this registry."""
        return self._name
