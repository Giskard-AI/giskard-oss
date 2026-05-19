from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from pydantic import BaseModel


class ScenarioGenerator(BaseModel):
    tags: ClassVar[list[str]] = []

    async def generate_scenario(
        self,
        description: str,
        languages: list[str],
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        raise NotImplementedError


_DATA_DIR = Path(__file__).parent / "data"


class DatasetScenarioGenerator(ScenarioGenerator):
    dataset_name: str

    async def generate_scenario(
        self,
        description: str,
        languages: list[str],
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        path = _DATA_DIR / f"{self.dataset_name}.jsonl"
        scenarios = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                scenario = Scenario.model_validate_json(line)
                scenario = scenario.with_annotations(
                    {
                        **scenario.annotations,
                        "description": description,
                        "languages": languages,
                    }
                )
                scenarios.append(scenario)

        if max_scenarios is not None and max_scenarios < len(scenarios):
            rng = rng or np.random.default_rng(42)
            indices = rng.choice(len(scenarios), size=max_scenarios, replace=False)
            scenarios = [scenarios[i] for i in sorted(indices)]

        return scenarios
