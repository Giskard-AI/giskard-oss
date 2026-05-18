from .adversarial_generator import AdversarialScenarioGenerator
from .base import ScenarioGenerator
from .prompt_injection import PromptInjectionScenarioGenerator


def _normalize_generator(
    generator: ScenarioGenerator | type[ScenarioGenerator],
) -> ScenarioGenerator:
    if isinstance(generator, type):
        return generator()
    return generator


class SuiteGeneratorRegistry:
    def __init__(self) -> None:
        self._generators: list[ScenarioGenerator] = []

    def register(self, generator: ScenarioGenerator | type[ScenarioGenerator]) -> None:
        instance = _normalize_generator(generator)
        if not isinstance(instance, ScenarioGenerator):
            raise TypeError(
                f"Expected a ScenarioGenerator instance or subclass, got {type(instance).__name__}"
            )
        if any(instance == existing for existing in self._generators):
            raise ValueError(
                f"{type(instance).__name__} is already registered with equivalent configuration"
            )
        self._generators.append(instance)

    def unregister(
        self, generator: ScenarioGenerator | type[ScenarioGenerator]
    ) -> None:
        instance = _normalize_generator(generator)
        for i, existing in enumerate(self._generators):
            if instance == existing:
                del self._generators[i]
                return
        raise ValueError(f"{type(instance).__name__} is not registered")

    def clear(self) -> None:
        self._generators.clear()

    def list(self) -> list[ScenarioGenerator]:
        return list(self._generators)


suite_generator_registry = SuiteGeneratorRegistry()
suite_generator_registry.register(AdversarialScenarioGenerator)
suite_generator_registry.register(PromptInjectionScenarioGenerator)
