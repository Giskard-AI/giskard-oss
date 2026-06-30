from typing import Any, Protocol

from giskard.checks import SuiteResult, Target


class ScanAdapter(Protocol):
    async def run(
        self,
        target: Target,  # pyright: ignore[reportMissingTypeArgument]
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> SuiteResult: ...
