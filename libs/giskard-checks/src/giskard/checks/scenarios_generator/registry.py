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
    """Global registry of :class:`~giskard.checks.scenarios_generator.base.ScenarioGenerator` instances.

    The module-level singleton :data:`suite_generator_registry` is pre-populated
    with the built-in generators.  Callers can extend it (e.g. in a plugin) via
    :meth:`register` and restore the original state via :meth:`unregister` or
    :meth:`clear`.

    Duplicate detection relies on Pydantic's structural ``__eq__``: two
    generator instances with identical configuration are considered the same.
    """

    def __init__(self) -> None:
        self._generators: list[ScenarioGenerator] = []

    def register(self, generator: ScenarioGenerator | type[ScenarioGenerator]) -> None:
        """Add *generator* to the registry.

        Args:
            generator: A :class:`~giskard.checks.scenarios_generator.base.ScenarioGenerator`
                instance or its subclass (instantiated with default settings).

        Raises:
            TypeError: If *generator* is not a :class:`ScenarioGenerator`.
            ValueError: If an equivalent generator is already registered.
        """
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
        """Remove *generator* from the registry.

        Args:
            generator: The instance or class to remove.  Matched by structural
                equality, so the argument does not have to be the exact object
                passed to :meth:`register`.

        Raises:
            ValueError: If no matching generator is found.
        """
        instance = _normalize_generator(generator)
        for i, existing in enumerate(self._generators):
            if instance == existing:
                del self._generators[i]
                return
        raise ValueError(f"{type(instance).__name__} is not registered")

    def clear(self) -> None:
        """Remove all registered generators."""
        self._generators.clear()

    def generators(self) -> list[ScenarioGenerator]:
        """Return a snapshot of the currently registered generators."""
        return list(self._generators)


suite_generator_registry = SuiteGeneratorRegistry()
suite_generator_registry.register(AdversarialScenarioGenerator)
suite_generator_registry.register(PromptInjectionScenarioGenerator)
