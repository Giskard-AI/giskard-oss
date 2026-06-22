from giskard.scan.generators.base import (
    DatasetScenarioGenerator,
    ScenarioContext,
    ScenarioGenerator,
)


class _Stub(ScenarioGenerator):
    async def generate_scenario(
        self, context: ScenarioContext, max_scenarios=None, rng=None
    ):
        return []


def test_scenario_generator_is_importable():
    gen = _Stub()
    assert isinstance(gen, ScenarioGenerator)


def test_dataset_generator_has_dataset_name_field():
    class _DS(DatasetScenarioGenerator):
        dataset_name: str = "stub"

    gen = _DS()
    assert gen.dataset_name == "stub"
