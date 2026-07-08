"""Resolve DeepTeam vulnerability/attack names to instantiated objects.

Name->class maps are the single source of truth for what this integration
supports. Single-turn vs multi-turn is decided by which map an attack lives in.
All ``deepteam`` imports are lazy so this module imports without deepteam
installed.
"""

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


def resolve_vulnerabilities(names: "list[str] | None") -> list[Any]:
    """Return instantiated DeepTeam vulnerability objects for *names*.

    ``None`` -> the curated default set. ``[]`` -> ``[]``.
    """
    classes = _vulnerability_classes()
    selected = DEFAULT_VULNERABILITIES if names is None else names
    return [_instantiate(name, classes, "vulnerability") for name in selected]


def resolve_attacks(names: "list[str] | None", *, singleturn: bool) -> list[Any]:
    """Return instantiated DeepTeam attack objects for *names*.

    ``None`` -> the curated default set. ``[]`` -> ``[]``. When ``singleturn`` is
    True, every attack whose class lives in ``deepteam.attacks.multi_turn`` is
    dropped (a multi-turn attack has no valid single-turn form).
    """
    single = _singleturn_attack_classes()
    multi = _multiturn_attack_classes()
    combined = {**single, **multi}
    selected = DEFAULT_ATTACKS if names is None else names
    resolved: list[Any] = []
    for name in selected:
        if singleturn and name in multi:
            continue
        resolved.append(_instantiate(name, combined, "attack"))
    return resolved


def _instantiate(name: str, classes: dict[str, type], kind: str) -> Any:
    try:
        cls = classes[name]
    except KeyError:
        valid = ", ".join(sorted(classes))
        raise ValueError(
            f"Unknown deepteam {kind} {name!r}. Valid names: {valid}"
        ) from None
    return cls()
