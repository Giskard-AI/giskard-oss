import asyncio
import inspect
from typing import Any

try:
    import garak.generators.base as _garak_base
    from garak.generators.base import (  # pyright: ignore[reportMissingImports]
        Message as _Message,
    )

    _garak_available = True
except ImportError:
    _garak_base = None  # type: ignore[assignment]
    _Message = None  # type: ignore[assignment]
    _garak_available = False


def _make_base() -> type:
    if _garak_available:
        return _garak_base.Generator  # type: ignore[union-attr]  # pyright: ignore[reportOptionalMemberAccess]
    return object


class GiskardGenerator(_make_base()):  # type: ignore[misc]
    """Wraps a Giskard Target as a garak Generator (garak 0.15+)."""

    name = "giskard"
    generator_family_name = "giskard"
    supports_multiple_generations = False
    extra_dependency_names: list[str] = []
    description = "Giskard target bridge"

    def __init__(self, target: Any) -> None:
        if not _garak_available:
            raise ImportError(
                "garak is not installed. Run: pip install giskard-scan[garak]"
            )
        self._giskard_target = target
        super().__init__(name="giskard")  # type: ignore[call-arg]

    def _call_model(  # type: ignore[override]
        self, prompt: Any, generations_this_call: int = 1
    ) -> list[Any]:
        # prompt is a Conversation in garak 0.15+
        text = (
            prompt.last_message().text
            if hasattr(prompt, "last_message")
            else str(prompt)
        )
        if inspect.iscoroutinefunction(self._giskard_target):
            response = asyncio.run(self._giskard_target(text))
        else:
            response = self._giskard_target(text)
        out = response if isinstance(response, str) else str(response)
        return [_Message(text=out)]  # type: ignore[misc]  # pyright: ignore[reportOptionalCall]
