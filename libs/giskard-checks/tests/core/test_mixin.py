from unittest.mock import MagicMock

import giskard.checks.settings as settings
from giskard.agents.generators.base import BaseGenerator
from giskard.checks.core.mixin import WithGeneratorMixin
from giskard.checks.settings import set_default_generator


class ConcreteCheck(WithGeneratorMixin):
    pass


def test_generator_reflects_global_change_after_instantiation():
    """Instance created before set_default_generator must see the new default."""
    check = ConcreteCheck()

    new_gen = MagicMock(spec=BaseGenerator)
    set_default_generator(new_gen)

    assert check._generator is new_gen


def test_explicit_generator_is_not_overridden():
    """Explicitly passed generator must be preserved even if global changes."""
    explicit_gen = MagicMock(spec=BaseGenerator)
    check = ConcreteCheck(generator=explicit_gen)

    other_gen = MagicMock(spec=BaseGenerator)
    set_default_generator(other_gen)

    assert check.generator is explicit_gen


def test_default_generator_is_returned_when_none_set():
    """When no global set and no explicit generator, must return a generator."""

    settings._default_generator = None
    check = ConcreteCheck()
    assert check._generator is not None
