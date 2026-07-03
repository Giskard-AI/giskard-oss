"""Top-level entry point for third-party scanner integrations."""

from typing import Any, Literal

from giskard.checks import SuiteResult, Target, Trace


async def third_party_scan[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    target: Target[InputType, OutputType, TraceType],
    tool: Literal["garak"] = "garak",
    **kwargs: Any,
) -> SuiteResult:
    """Run an external security scanner against a Giskard target.

    Args:
        target: Agent or provider target to evaluate.
        tool: Scanner to use. Only ``"garak"`` is supported today.
        **kwargs: Tool-specific options. For garak:
            ``probes: list[str] | None`` restricts which probes run; omitted
            means all active loadable probes, while an empty list runs none.
            ``target_mode: str`` (default ``"multiturn"``) skips garak
            iterative probes when set to ``"singleturn"``.

    Returns:
        The completed suite result.
    """
    if tool == "garak":
        from .garak import GarakScanAdapter

        adapter = GarakScanAdapter()
    else:
        raise ValueError(f"Unknown tool {tool!r}. Available: ['garak']")
    return await adapter.run(target, **kwargs)
