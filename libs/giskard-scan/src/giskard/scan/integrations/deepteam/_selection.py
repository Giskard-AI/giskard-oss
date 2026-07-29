"""Resolve DeepTeam vulnerability/attack names to instantiated objects.

Name->class maps are the single source of truth for what this integration
supports. Single-turn vs multi-turn is decided by which map an attack lives in.
Unknown or filtered names become skip markers (log + SuiteResult skips) rather
than raising, matching the garak adapter.
All ``deepteam`` imports are lazy so this module imports without deepteam
installed.
"""

from dataclasses import dataclass
from typing import Any

# Curated defaults (broad, balanced) used when the caller passes None.
DEFAULT_VULNERABILITIES: list[str] = [
    "Bias",
    "Toxicity",
    "PIILeakage",
    "PromptLeakage",
    "Misinformation",
]
DEFAULT_ATTACKS: list[str] = [
    "PromptInjection",
    "Roleplay",
    "Leetspeak",
    "LinearJailbreaking",
    "CrescendoJailbreaking",
]


@dataclass(frozen=True)
class SkipMarker:
    """A requested vulnerability/attack name that will not run."""

    name: str
    reason: str


def _vulnerability_classes() -> dict[str, type]:
    import deepteam.vulnerabilities as v

    names = [
        "Bias",
        "Toxicity",
        "PIILeakage",
        "PromptLeakage",
        "Misinformation",
    ]
    return {name: getattr(v, name) for name in names}


def _singleturn_attack_classes() -> dict[str, type]:
    import deepteam.attacks.single_turn as s

    names = ["PromptInjection", "Roleplay", "Leetspeak", "ROT13"]
    return {name: getattr(s, name) for name in names}


def _multiturn_attack_classes() -> dict[str, type]:
    import deepteam.attacks.multi_turn as m

    names = [
        "LinearJailbreaking",
        "CrescendoJailbreaking",
        "TreeJailbreaking",
        "SequentialJailbreak",
        "BadLikertJudge",
    ]
    return {name: getattr(m, name) for name in names}


def list_vulnerabilities() -> list[str]:
    """Return supported DeepTeam vulnerability names for this integration."""
    return list(_vulnerability_classes())


def list_attacks() -> list[str]:
    """Return supported DeepTeam attack names (single- and multi-turn)."""
    return sorted({*_singleturn_attack_classes(), *_multiturn_attack_classes()})


def resolve_vulnerabilities(
    names: "list[str] | None",
) -> tuple[list[Any], list[SkipMarker]]:
    """Return instantiated vulnerabilities plus skip markers for unknown names.

    ``None`` -> the curated default set. ``[]`` -> ``([], [])``.
    """
    classes = _vulnerability_classes()
    selected = DEFAULT_VULNERABILITIES if names is None else names
    resolved: list[Any] = []
    skipped: list[SkipMarker] = []
    for name in selected:
        if name not in classes:
            skipped.append(SkipMarker(name=name, reason="unknown"))
            continue
        resolved.append(classes[name]())
    return resolved, skipped


def resolve_attacks(
    names: "list[str] | None", *, singleturn: bool
) -> tuple[list[Any], list[SkipMarker]]:
    """Return instantiated attacks plus skip markers for unknown/filtered names.

    ``None`` -> the curated default set. ``[]`` -> ``([], [])``. When
    ``singleturn`` is True, multi-turn attacks are skipped (not raised).
    """
    single = _singleturn_attack_classes()
    multi = _multiturn_attack_classes()
    combined = {**single, **multi}
    selected = DEFAULT_ATTACKS if names is None else names
    resolved: list[Any] = []
    skipped: list[SkipMarker] = []
    for name in selected:
        if name not in combined:
            skipped.append(SkipMarker(name=name, reason="unknown"))
            continue
        if singleturn and name in multi:
            skipped.append(
                SkipMarker(
                    name=name,
                    reason="filtered by target_mode='singleturn'",
                )
            )
            continue
        resolved.append(combined[name]())
    return resolved, skipped
