from .base import ScenarioGenerator


def _normalize(
    generator: ScenarioGenerator | type[ScenarioGenerator],
) -> ScenarioGenerator:
    if isinstance(generator, type):
        return generator()
    return generator


class SuiteGeneratorRegistry:
    def __init__(self) -> None:
        self._generators: list[ScenarioGenerator] = []

    def register(self, generator: ScenarioGenerator | type[ScenarioGenerator]) -> None:
        instance = _normalize(generator)
        if any(instance == existing for existing in self._generators):
            raise ValueError(
                f"{type(instance).__name__} is already registered with equivalent configuration"
            )
        self._generators.append(instance)

    def unregister(
        self, generator: ScenarioGenerator | type[ScenarioGenerator]
    ) -> None:
        instance = _normalize(generator)
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
