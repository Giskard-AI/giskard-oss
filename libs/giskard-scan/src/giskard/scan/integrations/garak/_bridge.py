import asyncio
import inspect
from typing import Any

try:
    from garak.generators.base import (  # pyright: ignore[reportMissingImports]
        Generator as _GarakBase,
    )

    _garak_available = True
except ImportError:
    _GarakBase = object  # type: ignore[assignment,misc]
    _garak_available = False


class GiskardGenerator(_GarakBase):  # pyright: ignore[reportGeneralTypeIssues]
    """Wraps a Giskard Target as a garak Generator.

    Must be used inside a thread (e.g. via run_in_executor) so that
    asyncio.run() is safe for async targets.
    """

    supports_multiple_generations = False

    def __init__(self, target: Any) -> None:  # pyright: ignore[reportMissingSuperCall]
        if not _garak_available:
            raise ImportError(
                "garak is not installed. Run: pip install giskard-scan[garak]"
            )
        self._giskard_target = target
        # ponytail: bypass parent __init__ (heavy model-loading setup)
        self.name = "giskard"
        self.generations = 1
        self.temperature = None
        self.max_tokens = None

    def _call_model(
        self, prompt: str, generations_this_call: int = 1
    ) -> list[str | None]:
        if inspect.iscoroutinefunction(self._giskard_target):
            response = asyncio.run(self._giskard_target(prompt))
        else:
            response = self._giskard_target(prompt)
        return [response if isinstance(response, str) else str(response)]
