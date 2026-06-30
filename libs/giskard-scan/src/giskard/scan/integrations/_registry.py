from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._adapter import ScanAdapter

_REGISTRY: dict[str, type["ScanAdapter"]] = {}


def register(name: str, adapter_class: type["ScanAdapter"]) -> None:
    _REGISTRY[name] = adapter_class


def get(name: str) -> type["ScanAdapter"] | None:
    return _REGISTRY.get(name)


def available() -> list[str]:
    return list(_REGISTRY.keys())
