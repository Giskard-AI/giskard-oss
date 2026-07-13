"""Top-level entry point for third-party scanner integrations.

Lidar is a private Giskard package, not a public integration: ``tool="lidar"``
stays dispatchable for callers who already have it, but it is deliberately left
out of the public docstring, the unknown-tool error message, and the README.
Keep it that way -- do not "complete" the docs by adding it back.
"""

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

    Requires the scanner's optional extra: ``pip install giskard-scan[garak]``
    or ``pip install giskard-scan[deepteam]``.

    Args:
        target: Agent or provider target to evaluate.
        tool: Scanner to use, ``"garak"`` or ``"deepteam"``.
        description: Natural-language description of the agent under test.
            Deepteam uses it as ``red_team``'s ``target_purpose``; garak has no
            target-profile concept and ignores it.
        languages: BCP-47 language codes the agent handles. Ignored by garak and
            deepteam.
        **kwargs: Tool-specific options. For garak: ``probes: list[str] | None``
            (None runs every active probe) and ``target_mode: str`` (default
            ``"multiturn"``; ``"singleturn"`` drops garak's iterative probes).
            For deepteam: ``vulnerabilities: list[str] | None`` and ``attacks:
            list[str] | None`` (name lists; None runs a curated default set), and
            ``target_mode: str`` (default ``"multiturn"``) which drops
            multi-turn attacks when set to ``"singleturn"``.

    Returns:
        The completed suite result.

    Raises:
        ImportError: The selected scanner's optional extra is not installed.
        ValueError: ``tool`` is unknown, or a deepteam vulnerability/attack name
            is not recognized.
        TypeError: A keyword argument is not valid for the selected tool.
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
        raise ValueError(f"Unknown tool {tool!r}. Available: ['garak', 'deepteam']")
