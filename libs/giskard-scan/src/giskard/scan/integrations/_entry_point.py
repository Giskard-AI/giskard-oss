"""Top-level entry point for third-party scanner integrations."""

from typing import Any, Literal, overload

from giskard.checks import SuiteResult, Target, Trace


@overload
async def third_party_scan[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    target: Target[InputType, OutputType, TraceType],
    tool: Literal["garak"],
    *,
    description: str,
    languages: list[str] | None = None,
    probes: list[str] | None = None,
    target_mode: str = "multiturn",
) -> SuiteResult: ...


@overload
async def third_party_scan[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    target: Target[InputType, OutputType, TraceType],
    tool: Literal["lidar"],
    *,
    description: str,
    languages: list[str] | None = None,
    probes: list[str] | None = None,
    tags: list[str] | None = None,
    target_mode: str = "multiturn",
) -> SuiteResult: ...


@overload
async def third_party_scan[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    target: Target[InputType, OutputType, TraceType],
    tool: Literal["deepteam"],
    *,
    description: str,
    languages: list[str] | None = None,
    vulnerabilities: list[str] | None = None,
    attacks: list[str] | None = None,
    target_mode: str = "multiturn",
) -> SuiteResult: ...


async def third_party_scan[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    target: Target[InputType, OutputType, TraceType],
    tool: Literal["garak", "lidar", "deepteam"],
    *,
    description: str,
    languages: list[str] | None = None,
    **kwargs: Any,
) -> SuiteResult:
    """Run an external security scanner against a Giskard target.

    Args:
        target: Agent or provider target to evaluate.
        tool: Scanner to use. ``"garak"``, ``"lidar"``, or ``"deepteam"``.
        description: Natural-language description of the agent under test.
            For lidar this becomes the ``TargetInfo.agent_description``; for
            deepteam it becomes ``red_team``'s ``target_purpose``. Garak has no
            target-profile concept and ignores it.
        languages: BCP-47 language codes the agent handles. Used by lidar;
            ignored by garak and deepteam.
        **kwargs: Tool-specific options. For garak: ``probes``, ``target_mode``.
            For lidar: ``probes``, ``tags``, ``target_mode``. For deepteam:
            ``vulnerabilities: list[str] | None`` and ``attacks: list[str] |
            None`` (name lists; None runs a curated default set), and
            ``target_mode: str`` (default ``"multiturn"``) which drops
            multi-turn attacks when set to ``"singleturn"``.

    Returns:
        The completed suite result.
    """
    if tool == "garak":
        from .garak import GarakScanAdapter

        return await GarakScanAdapter().run(target, **kwargs)
    elif tool == "lidar":
        from .lidar import LidarScanAdapter

        return await LidarScanAdapter().run(
            target, description=description, languages=languages, **kwargs
        )
    elif tool == "deepteam":
        from .deepteam import DeepTeamScanAdapter

        return await DeepTeamScanAdapter().run(
            target, description=description, languages=languages, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown tool {tool!r}. Available: ['garak', 'lidar', 'deepteam']"
        )
