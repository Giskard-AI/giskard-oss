"""generate_suite checkpoint / resume tests."""

from pathlib import Path
from typing import Any

import numpy as np
from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from giskard.checks.utils.checkpoint import (
    generate_fingerprint,
    generator_checkpoint_key,
    store_path_for,
)
from giskard.scan.catalog import generate_suite
from giskard.scan.generators.base import ScenarioContext, ScenarioGenerator


class _CountingGenerator(ScenarioGenerator):
    """Returns a fixed number of scenarios and records call count."""

    name: str = "counting"
    scenario_count: int = 2
    calls: int = 0

    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
        target_mode: str = "multiturn",
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        self.calls += 1
        n = max_scenarios if max_scenarios is not None else self.scenario_count
        return [Scenario(name=f"{self.name}-{i}") for i in range(n)]


def _generate_store_path(
    root: Path,
    *,
    generators: list[_CountingGenerator],
    seed: int = 42,
) -> Path:
    keys = [
        generator_checkpoint_key(generator, index)
        for index, generator in enumerate(generators)
    ]
    fingerprint = generate_fingerprint(
        description="bot",
        languages=["en"],
        generator_keys=keys,
        seed=seed,
        target_mode="multiturn",
        max_scenarios=None,
    )
    return store_path_for(root, fingerprint)


async def test_generate_suite_writes_checkpoint(tmp_path: Path) -> None:
    g1 = _CountingGenerator(name="one", scenario_count=1)
    g2 = _CountingGenerator(name="two", scenario_count=1)
    suite = await generate_suite(
        "bot",
        languages=["en"],
        generators=[g1, g2],
        checkpoint_dir=tmp_path / "ck",
    )
    assert len(suite.scenarios) == 2
    events = (
        (_generate_store_path(tmp_path / "ck", generators=[g1, g2]) / "events.jsonl")
        .read_text()
        .strip()
        .splitlines()
    )
    types = {line.split('"type": "')[1].split('"')[0] for line in events}
    assert "scenario_generated" in types
    assert "generator_completed" in types


async def test_generate_suite_resume_skips_finished_generators(tmp_path: Path) -> None:
    ck = tmp_path / "ck"
    g1 = _CountingGenerator(name="one", scenario_count=1)
    g2 = _CountingGenerator(name="two", scenario_count=1)
    first = await generate_suite(
        "bot",
        languages=["en"],
        generators=[g1, g2],
        checkpoint_dir=ck,
        seed=7,
    )
    assert len(first.scenarios) == 2
    assert g1.calls == 1 and g2.calls == 1

    g1b = _CountingGenerator(name="one", scenario_count=1)
    g2b = _CountingGenerator(name="two", scenario_count=1)
    second = await generate_suite(
        "bot",
        languages=["en"],
        generators=[g1b, g2b],
        checkpoint_dir=ck,
        seed=7,
    )
    assert len(second.scenarios) == 2
    assert g1b.calls == 0 and g2b.calls == 0
    assert {s.name for s in second.scenarios} == {s.name for s in first.scenarios}
