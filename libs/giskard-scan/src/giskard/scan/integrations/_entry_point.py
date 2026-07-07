"""Top-level entry point for third-party scanner integrations."""

from typing import Any, Literal

from giskard.checks import SuiteResult, Target, Trace


async def third_party_scan[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    target: Target[InputType, OutputType, TraceType],
    tool: Literal["garak", "lidar"],
    *,
    description: str,
    languages: list[str] | None = None,
    **kwargs: Any,
) -> SuiteResult:
    """Run an external security scanner against a Giskard target.

    Args:
        target: Agent or provider target to evaluate.
        tool: Scanner to use. ``"garak"`` or ``"lidar"``.
        description: Natural-language description of the agent under test.
            For lidar this becomes the ``TargetInfo.agent_description`` that
            probes use to build their attacks. Garak has no target-profile
            concept and ignores it.
        languages: BCP-47 language codes the agent handles. For lidar this
            becomes ``TargetInfo.languages``. Garak ignores it.
        **kwargs: Tool-specific options. For garak:
            ``probes: list[str] | None`` restricts which probes run; omitted
            means all active loadable probes, while an empty list runs none.
            ``target_mode: str`` (default ``"multiturn"``) skips garak
            iterative probes when set to ``"singleturn"``.
          For lidar:
            ``probes: list[str] | None`` restricts which probes run by id
            (e.g. ``"deepset-injection:1.0"``); ``None`` runs all.
            ``tags: list[str] | None`` restricts probes by tag; ``None`` runs
            all. ``target_mode: str`` (default ``"multiturn"``) skips lidar's
            multi-turn probes (crescendo, goat, ...) when set to
            ``"singleturn"``.

    Returns:
        The completed suite result.
    """
    if tool == "garak":
        from .garak import GarakScanAdapter

        # Garak has no TargetInfo concept: drop the context args so its
        # run() signature is unaffected.
        return await GarakScanAdapter().run(target, **kwargs)
    elif tool == "lidar":
        from .lidar import LidarScanAdapter

        return await LidarScanAdapter().run(
            target, description=description, languages=languages, **kwargs
        )
    else:
        raise ValueError(f"Unknown tool {tool!r}. Available: ['garak', 'lidar']")
