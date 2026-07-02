"""Third-party scanner integrations for giskard.scan (experimental)."""

from typing import Any, Protocol

from giskard.checks import SuiteResult, Target, Trace

from ._entry_point import third_party_scan


class ScanAdapter(Protocol):
    async def run[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
        self,
        target: Target[InputType, OutputType, TraceType],
        **kwargs: Any,
    ) -> SuiteResult: ...


__all__ = ["ScanAdapter", "third_party_scan"]
