from .generators.base import ScenarioGenerator


def _normalize_generator(
    generator: "ScenarioGenerator | type[ScenarioGenerator]",
) -> ScenarioGenerator:
    if isinstance(generator, type):
        return generator()
    return generator


class SuiteGeneratorRegistry:
    """Mutable registry of scenario generator instances."""

    def __init__(self) -> None:
        self._generators: list[ScenarioGenerator] = []

    def register(
        self, generator: "ScenarioGenerator | type[ScenarioGenerator]"
    ) -> None:
        """Add a generator to the registry.

        Parameters
        ----------
        generator : ScenarioGenerator or type
            Generator instance, or a subclass to instantiate with its defaults.

        Raises
        ------
        TypeError
            If it is not a :class:`ScenarioGenerator`.
        ValueError
            If an equivalently configured generator is already registered.
        """
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
        """Remove a previously registered generator.

        Parameters
        ----------
        generator : ScenarioGenerator or type
            Generator to remove; a subclass matches an instance with defaults.

        Raises
        ------
        ValueError
            If the generator is not registered.
        """
        instance = _normalize_generator(generator)
        try:
            self._generators.remove(instance)
        except ValueError:
            raise ValueError(f"{type(instance).__name__} is not registered") from None

    def clear(self) -> None:
        """Remove every registered generator."""
        self._generators.clear()

    def generators(self, commercial_use: bool = False) -> list[ScenarioGenerator]:
        """Return the registered generators.

        Parameters
        ----------
        commercial_use : bool, default False
            Keep only generators whose data allows commercial use.

        Returns
        -------
        list of ScenarioGenerator
            Generators in registration order.
        """
        return [
            generator
            for generator in self._generators
            if not commercial_use or generator.allow_commercial_use
        ]
