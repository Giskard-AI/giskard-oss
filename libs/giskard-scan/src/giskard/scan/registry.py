from .generators.adversarial import AdversarialScenarioGenerator
from .generators.base import ScenarioGenerator
from .generators.goat import GOATAttackScenarioGenerator
from .generators.prompt_injection import PromptInjectionScenarioGenerator


def _normalize_generator(
    generator: "ScenarioGenerator | type[ScenarioGenerator]",
) -> ScenarioGenerator:
    if isinstance(generator, type):
        return generator()
    return generator


class SuiteGeneratorRegistry:
    """Global registry of ScenarioGenerator instances."""

    def __init__(self) -> None:
        self._generators: list[ScenarioGenerator] = []

    def register(
        self, generator: "ScenarioGenerator | type[ScenarioGenerator]"
    ) -> None:
        instance = _normalize_generator(generator)
        if not isinstance(instance, ScenarioGenerator):
            raise TypeError(
                f"Expected a ScenarioGenerator instance or subclass, got {type(instance).__name__}"
            )
        if instance in self._generators:
            raise ValueError(
                f"{type(instance).__name__} is already registered with equivalent configuration"
            )
        self._generators.append(instance)

    def unregister(
        self, generator: "ScenarioGenerator | type[ScenarioGenerator]"
    ) -> None:
        instance = _normalize_generator(generator)
        try:
            self._generators.remove(instance)
        except ValueError:
            raise ValueError(f"{type(instance).__name__} is not registered") from None

    def clear(self) -> None:
        self._generators.clear()

    def generators(self) -> list[ScenarioGenerator]:
        return list(self._generators)


suite_generator_registry = SuiteGeneratorRegistry()
suite_generator_registry.register(AdversarialScenarioGenerator)
suite_generator_registry.register(GOATAttackScenarioGenerator)
suite_generator_registry.register(PromptInjectionScenarioGenerator)
