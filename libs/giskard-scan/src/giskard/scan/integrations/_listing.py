"""Shared discovery helpers across native and third-party scan tools."""

from typing import Literal

type ScanTool = Literal["giskard", "garak", "deepteam"]


def _list_giskard_generators() -> list[str]:
    from ..quality import quality_suite_generator_registry
    from ..vulnerability import vulnerability_suite_generator_registry

    names = {
        type(generator).__name__
        for generator in vulnerability_suite_generator_registry.generators()
    }
    names.update(
        type(generator).__name__
        for generator in quality_suite_generator_registry.generators()
    )
    return sorted(names)


def list_scan_items(
    tool: ScanTool,
    *,
    include_inactive: bool = False,
) -> list[str]:
    """List selectable item names for a scan tool.

    Args:
        tool: ``"giskard"`` returns scenario generator class names,
            ``"garak"`` returns probe plugin names, ``"deepteam"`` returns
            supported vulnerability and attack names. Other values raise
            ``ValueError``.
        include_inactive: Garak only — also include inactive catalog probes.
            Ignored for other tools.

    Returns:
        Sorted unique names suitable for the corresponding scan API.

    Raises:
        ValueError: If ``tool`` is not one of the supported values.
        ImportError: If an optional tool dependency is not installed.
    """
    if tool == "giskard":
        return _list_giskard_generators()
    if tool == "garak":
        from .garak import list_probes

        return list_probes(include_inactive=include_inactive)
    if tool == "deepteam":
        from .deepteam import list_attacks, list_vulnerabilities

        return sorted({*list_vulnerabilities(), *list_attacks()})
    raise ValueError(
        f"Unknown tool {tool!r}. Available: ['giskard', 'garak', 'deepteam']"
    )
