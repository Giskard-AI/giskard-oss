import random
from enum import Enum
from pathlib import Path
from typing import Any

from ..core.interaction import Trace
from ..core.scenario import Scenario
from .suite import Suite

_DATA_DIR = Path(__file__).parent / "data"


class ScenarioCategory(str, Enum):
    """Scenario categories available for suite generation."""

    LLM01_INDIRECT_INJECTION = "llm01_indirect_injection"


def _load_scenarios(
    category: ScenarioCategory,
) -> list[Scenario[str, Any, Trace[str, Any]]]:
    path = _DATA_DIR / f"{category.value}.jsonl"
    scenarios = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scenario = Scenario.model_validate_json(line)
            scenarios.append(scenario)
    return scenarios


def generate_suite(
    description: str,
    categories: list[ScenarioCategory] | None = None,
    max_scenarios: int | None = None,
    seed: int = 42,
    name: str = "Security Suite",
) -> Suite[str, Any]:
    """Generate a Suite of scenarios for the given categories.

    Parameters
    ----------
    description : str
        Description of the agent under test. Injected into each scenario's
        annotations as ``"description"``; overwrites any existing value from
        the JSONL so prompt templates can adapt to the agent's context.
    categories : list[ScenarioCategory] | None
        Categories to include. ``None`` (default) selects all available categories.
    max_scenarios : int | None
        Maximum number of scenarios to include. None means all.
    seed : int
        Random seed for reproducible sampling (default: 42).
    name : str
        Suite name (default: "Security Suite").

    Returns
    -------
    Suite
        A Suite with all selected scenarios, no target bound.
    """
    selected = categories if categories is not None else list(ScenarioCategory)
    all_scenarios: list[Scenario[str, Any, Trace[str, Any]]] = []
    for category in selected:
        all_scenarios.extend(_load_scenarios(category))

    if max_scenarios is not None:
        rng = random.Random(seed)
        all_scenarios = rng.sample(
            all_scenarios, min(max_scenarios, len(all_scenarios))
        )

    for scenario in all_scenarios:
        scenario.annotations = {**scenario.annotations, "description": description}

    return Suite(name=name, scenarios=all_scenarios)
