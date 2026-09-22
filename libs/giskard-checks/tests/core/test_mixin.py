from unittest.mock import MagicMock

import giskard.checks.settings as settings
import pytest
from giskard.agents import BaseEmbeddingModel, BaseGenerator, Generator
from giskard.checks import BaseJudge, LLMChatJudge, LLMGenerator, LLMJudge
from giskard.checks.core.mixin import WithEmbeddingMixin, WithGeneratorMixin
from giskard.checks.settings import (
    set_default_embedding_model,
    set_default_generator,
    set_default_judge,
)


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


def test_judge_reflects_global_change_after_instantiation():
    check = LLMJudge(prompt="Evaluate the answer.")
    judge = LLMChatJudge(generator=Generator(model="openai/gpt-4o-mini"))

    set_default_judge(judge)

    assert check._judge is judge
    assert check.judge is None


def test_judge_and_input_generator_resolve_separate_defaults():
    check = LLMJudge(prompt="Evaluate the answer.")
    input_generator = LLMGenerator(prompt="Ask a question.")
    generation = Generator(model="azure_ai/gpt-5.6-luna")
    judge = LLMChatJudge(generator=Generator(model="openai/gpt-4o-mini"))
    set_default_generator(generation)

    assert isinstance(check._judge, LLMChatJudge)
    assert check._judge._generator is generation
    assert input_generator._generator is generation

    set_default_judge(judge)

    assert check._judge is judge
    assert input_generator._generator is generation

    set_default_judge(None)

    assert isinstance(check._judge, LLMChatJudge)
    assert check._judge._generator is generation
    assert input_generator._generator is generation


def test_explicit_judge_generator_is_preserved():
    explicit = Generator(model="openai/gpt-4o-mini")
    check = LLMJudge(prompt="Evaluate the answer.", generator=explicit)

    set_default_generator("azure_ai/gpt-5.6-luna")
    set_default_judge("typesafe/jev")

    assert isinstance(check.judge, LLMChatJudge)
    assert isinstance(check._judge, LLMChatJudge)
    assert check.judge.generator is explicit
    assert check._judge.generator is explicit


def test_generator_and_judge_together_are_rejected():
    with pytest.raises(ValueError, match="both 'generator' and 'judge'"):
        LLMJudge(
            prompt="Evaluate the answer.",
            generator=Generator(model="openai/gpt-4o-mini"),
            judge=BaseJudge.model_validate("typesafe/jev"),
        )


def test_embedding_reflects_global_change_after_instantiation():
    embedding_user = WithEmbeddingMixin()
    embedding = MagicMock(spec=BaseEmbeddingModel)

    set_default_embedding_model(embedding)

    assert embedding_user._embedding_model is embedding
    assert embedding_user.embedding_model is None


def test_explicit_embedding_model_is_preserved():
    explicit = MagicMock(spec=BaseEmbeddingModel)
    embedding_user = WithEmbeddingMixin(embedding_model=explicit)

    set_default_embedding_model("text-embedding-3-large")

    assert embedding_user._embedding_model is explicit
