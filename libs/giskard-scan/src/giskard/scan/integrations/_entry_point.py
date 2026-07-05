"""Top-level entry point for third-party scanner integrations."""

from typing import Any, Literal

from giskard.checks import SuiteResult, Target, Trace


async def third_party_scan[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    target: Target[InputType, OutputType, TraceType],
    tool: Literal["garak", "lidar"] = "garak",
    **kwargs: Any,
) -> SuiteResult:
    """Run an external security scanner against a Giskard target.

    Args:
        target: Agent or provider target to evaluate.
        tool: Scanner to use. ``"garak"`` or ``"lidar"``.
        **kwargs: Tool-specific options. For garak:
            ``probes: list[str] | None`` restricts which probes run; omitted
            means all active loadable probes, while an empty list runs none.
            ``target_mode: str`` (default ``"multiturn"``) skips garak
            iterative probes when set to ``"singleturn"``.
          For lidar:
            ``probes: list[str] | None`` restricts which probes run by id
            (e.g. ``"deepset-injection:1.0"``); ``None`` runs all.
            ``tags: list[str] | None`` restricts probes by tag; ``None`` runs
            all. ``target_mode`` is ignored. Note: lidar probes generally need
            a discovered target profile, which this integration does not enable,
            so many probes report SKIP.

    Returns:
        The completed suite result.
    """
    if tool == "garak":
        from .garak import GarakScanAdapter

        adapter = GarakScanAdapter()
    elif tool == "lidar":
        from .lidar import LidarScanAdapter

        adapter = LidarScanAdapter()
    else:
        raise ValueError(f"Unknown tool {tool!r}. Available: ['garak', 'lidar']")
    return await adapter.run(target, **kwargs)
