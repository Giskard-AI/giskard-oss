from typing import Any

import numpy as np
from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from giskard.checks.scenarios_generator.base import DatasetScenarioGenerator


class _StubDatasetGenerator(DatasetScenarioGenerator):
    dataset_name: str = "stub"


def _make_scenarios(n: int) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
    return [Scenario(name=f"s{i}") for i in range(n)]


async def test_dataset_generator_no_budget_returns_all(tmp_path, monkeypatch):
    """With max_scenarios=None the generator returns every scenario."""
    import json

    import giskard.checks.scenarios_generator.base as base_mod

    stub_file = tmp_path / "stub.jsonl"
    stub_file.write_text(
        "\n".join(
            json.dumps(
                {
                    "name": f"s{i}",
                    "steps": [],
                    "annotations": {},
                }
            )
            for i in range(5)
        )
        + "\n"
    )
    monkeypatch.setattr(base_mod, "_DATA_DIR", tmp_path)
    gen = _StubDatasetGenerator()
    result = await gen.generate_scenario("desc", ["en"])
    assert len(result) == 5


async def test_dataset_generator_budget_subsamples(tmp_path, monkeypatch):
    """With max_scenarios=2, only 2 scenarios are returned."""
    import json

    import giskard.checks.scenarios_generator.base as base_mod

    stub_file = tmp_path / "stub.jsonl"
    stub_file.write_text(
        "\n".join(
            json.dumps({"name": f"s{i}", "steps": [], "annotations": {}})
            for i in range(10)
        )
        + "\n"
    )
    monkeypatch.setattr(base_mod, "_DATA_DIR", tmp_path)
    rng = np.random.default_rng(42)
    gen = _StubDatasetGenerator()
    result = await gen.generate_scenario("desc", ["en"], max_scenarios=2, rng=rng)
    assert len(result) == 2


async def test_dataset_generator_budget_larger_than_dataset_returns_all(
    tmp_path, monkeypatch
):
    """With max_scenarios > dataset size, all scenarios are returned."""
    import json

    import giskard.checks.scenarios_generator.base as base_mod

    stub_file = tmp_path / "stub.jsonl"
    stub_file.write_text(
        "\n".join(
            json.dumps({"name": f"s{i}", "steps": [], "annotations": {}})
            for i in range(3)
        )
        + "\n"
    )
    monkeypatch.setattr(base_mod, "_DATA_DIR", tmp_path)
    rng = np.random.default_rng(42)
    gen = _StubDatasetGenerator()
    result = await gen.generate_scenario("desc", ["en"], max_scenarios=100, rng=rng)
    assert len(result) == 3


async def test_dataset_generator_budget_reproducible(tmp_path, monkeypatch):
    """Same seed always picks the same subset."""
    import json

    import giskard.checks.scenarios_generator.base as base_mod

    stub_file = tmp_path / "stub.jsonl"
    stub_file.write_text(
        "\n".join(
            json.dumps({"name": f"s{i}", "steps": [], "annotations": {}})
            for i in range(10)
        )
        + "\n"
    )
    monkeypatch.setattr(base_mod, "_DATA_DIR", tmp_path)
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    gen = _StubDatasetGenerator()
    result_a = await gen.generate_scenario("desc", ["en"], max_scenarios=3, rng=rng_a)
    result_b = await gen.generate_scenario("desc", ["en"], max_scenarios=3, rng=rng_b)
    assert [s.name for s in result_a] == [s.name for s in result_b]
