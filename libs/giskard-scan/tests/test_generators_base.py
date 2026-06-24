from giskard.scan.generators.base import (
    LocalDatasetScenarioGenerator,
    ScenarioContext,
    ScenarioGenerator,
)


class _Stub(ScenarioGenerator):
    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios=None,
        rng=None,
        target_mode="multiturn",
    ):
        return []


def test_scenario_generator_is_importable():
    gen = _Stub()
    assert isinstance(gen, ScenarioGenerator)


async def test_scenario_generator_generate_scenario_accepts_target_mode():
    """generate_scenario must accept target_mode without raising TypeError."""
    from giskard.scan.generators.base import ScenarioGenerator

    class _Stub(ScenarioGenerator):
        async def generate_scenario(
            self,
            context: ScenarioContext,
            max_scenarios=None,
            rng=None,
            target_mode="multiturn",
        ):
            return []

    gen = _Stub()
    context = ScenarioContext(description="desc", languages=["en"])
    result = await gen.generate_scenario(context, target_mode="singleturn")
    assert result == []


def test_dataset_generator_has_dataset_name_field():
    class _DS(LocalDatasetScenarioGenerator):
        dataset_name: str = "stub"

    gen = _DS()
    assert gen.dataset_name == "stub"
