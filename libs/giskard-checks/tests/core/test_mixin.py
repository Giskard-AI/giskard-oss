from typing import Any
from unittest.mock import MagicMock

import giskard.checks.settings as settings
import pytest
from giskard.agents import (
    BaseEmbeddingModel,
    BaseGenerator,
    GenerationParams,
    Generator,
)
from giskard.checks import (
    BaseJudge,
    Check,
    Conformity,
    LLMChatJudge,
    LLMGenerator,
    LLMJudge,
    SOMJudge,
    Trace,
)
from giskard.checks.core.mixin import WithEmbeddingMixin, WithGeneratorMixin
from giskard.checks.settings import (
    set_default_embedding_model,
    set_default_generator,
    set_default_judge,
)
from pydantic import Field, ValidationError, create_model


class ConcreteCheck(WithGeneratorMixin):
    pass


class CustomGeneratorLLMJudge(LLMJudge[Any, Any, Trace[Any, Any]]):
    generator: BaseGenerator | None = Generator(model="openai/gpt-4o-mini")


class FactoryGeneratorLLMJudge(LLMJudge[Any, Any, Trace[Any, Any]]):
    generator: BaseGenerator | None = Field(
        default_factory=lambda: Generator(model="openai/gpt-4o-mini")
    )


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
    assert check.generator is explicit
    assert isinstance(check._judge, LLMChatJudge)
    assert check.judge.generator is explicit
    assert check._judge.generator is explicit


def test_class_level_generator_default_is_migrated_to_judge():
    check = CustomGeneratorLLMJudge(prompt="Evaluate the answer.")

    assert isinstance(check.judge, LLMChatJudge)
    assert check.generator is check.judge.generator
    assert isinstance(check.generator, Generator)
    assert check.generator.model == "openai/gpt-4o-mini"

    dumped = check.model_dump()

    assert "generator" not in dumped
    assert dumped["judge"]["generator"]["model"] == "openai/gpt-4o-mini"

    loaded = CustomGeneratorLLMJudge.model_validate(dumped)

    assert isinstance(loaded.judge, LLMChatJudge)
    assert loaded.generator is loaded.judge.generator
    assert isinstance(loaded.generator, Generator)
    assert loaded.generator.model == "openai/gpt-4o-mini"


@pytest.mark.parametrize("default_kind", ["value", "factory", "required"])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("json_mode", [False, True])
def test_nonnullable_generator_round_trip(
    default_kind: str, override: bool, json_mode: bool
):
    default = Generator(model="openai/gpt-4o-mini")
    factory = MagicMock(return_value=default)
    generator_field = {
        "value": default,
        "factory": Field(default_factory=factory),
        "required": ...,
    }[default_kind]
    configured_judge = create_model(
        "NonNullableGeneratorLLMJudge",
        __base__=LLMJudge[Any, Any, Trace[Any, Any]],
        generator=(BaseGenerator, generator_field),
    )
    selected = (
        Generator(
            model="openai/gpt-4o",
            params=GenerationParams(temperature=0.2, max_tokens=321),
        )
        if override
        else default
    )
    if override or default_kind == "required":
        check = configured_judge(prompt="Evaluate the answer.", generator=selected)
    else:
        check = configured_judge(prompt="Evaluate the answer.")
    factory.reset_mock()

    if json_mode:
        loaded = configured_judge.model_validate_json(check.model_dump_json())
    else:
        dumped = check.model_dump()
        loaded = configured_judge.model_validate(dumped)
        assert "generator" not in dumped

    assert isinstance(loaded.judge, LLMChatJudge)
    assert isinstance(loaded.generator, BaseGenerator)
    assert loaded.generator is loaded.judge.generator
    assert loaded.generator.model_dump() == selected.model_dump()
    assert loaded.model_dump() == check.model_dump()
    factory.assert_not_called()


def test_nonnullable_generator_round_trip_through_check_registry():
    configured_check = Check.register("nonnullable_generator_round_trip")(
        create_model(
            "NonNullableGeneratorConformity",
            __base__=Conformity,
            generator=(BaseGenerator, Generator(model="openai/gpt-4o-mini")),
        )
    )
    check = configured_check(rule="Be helpful.")

    for loaded in (
        Check.model_validate(check.model_dump()),
        Check.model_validate_json(check.model_dump_json()),
    ):
        assert isinstance(loaded, configured_check)
        assert loaded.model_dump() == check.model_dump()
        assert isinstance(loaded.judge, LLMChatJudge)
        assert loaded.generator is loaded.judge.generator


def test_explicit_none_clears_class_level_generator_default():
    check = CustomGeneratorLLMJudge(prompt="Evaluate the answer.", generator=None)

    assert check.generator is None
    assert check.judge is None


def test_required_class_level_generator_remains_required():
    required_generator_judge = create_model(
        "RequiredGeneratorLLMJudge",
        __base__=LLMJudge[Any, Any, Trace[Any, Any]],
        generator=(BaseGenerator | None, ...),
    )

    with pytest.raises(ValidationError, match="generator"):
        required_generator_judge(prompt="Evaluate the answer.")

    explicit = Generator(model="openai/gpt-4o-mini")
    check = required_generator_judge(prompt="Evaluate the answer.", generator=explicit)

    assert check.generator is explicit

    som_check = required_generator_judge(
        prompt="Evaluate the answer.", judge=BaseJudge.parse("typesafe/jev")
    )

    assert isinstance(som_check.judge, SOMJudge)
    assert som_check.generator is None


def test_class_level_generator_factory_is_migrated_to_judge():
    first = FactoryGeneratorLLMJudge(prompt="Evaluate the answer.")
    second = FactoryGeneratorLLMJudge(prompt="Evaluate the answer.")

    assert isinstance(first.generator, Generator)
    assert first.generator.model == "openai/gpt-4o-mini"
    assert first.generator is not second.generator


def test_class_level_data_factory_is_rejected_clearly():
    data_factory_judge = create_model(
        "DataFactoryGeneratorLLMJudge",
        __base__=LLMJudge[Any, Any, Trace[Any, Any]],
        generator=(
            BaseGenerator | None,
            Field(
                default_factory=lambda data: Generator(model=f"openai/{data['prompt']}")
            ),
        ),
    )

    with pytest.raises(TypeError, match="takes validated data"):
        data_factory_judge(prompt="gpt-4o-mini")


def test_generator_and_judge_together_are_rejected():
    with pytest.raises(ValueError, match="both 'generator' and 'judge'"):
        LLMJudge(
            prompt="Evaluate the answer.",
            generator=Generator(model="openai/gpt-4o-mini"),
            judge=BaseJudge.parse("typesafe/jev"),
        )


def test_explicit_none_generator_is_accepted():
    check = LLMJudge(prompt="Evaluate the answer.", generator=None)

    assert check.generator is None
    assert check.judge is None


def test_legacy_generator_is_excluded_from_dump():
    explicit = Generator(model="openai/gpt-4o-mini")
    check = LLMJudge(prompt="Evaluate the answer.", generator=explicit)

    dumped = check.model_dump()

    assert "generator" not in dumped
    assert dumped["judge"]["kind"] == "llm"
    assert dumped["judge"]["generator"]["model"] == "openai/gpt-4o-mini"


@pytest.mark.parametrize(
    "judge",
    [
        "typesafe/jev",
        '{"kind": "som", "model": {"kind": "typesafe", "model": "jev"}}',
    ],
)
def test_check_construction_accepts_loose_judge(judge: str):
    check = LLMJudge.model_validate({"prompt": "Evaluate the answer.", "judge": judge})

    assert isinstance(check.judge, BaseJudge)
    assert check.judge.kind == "som"


def test_post_init_generator_assignment_remigrates_to_judge():
    check = LLMJudge(prompt="Evaluate the answer.")
    explicit = Generator(model="openai/gpt-4o-mini")

    check.generator = explicit

    assert check.generator is explicit
    assert isinstance(check.judge, LLMChatJudge)
    assert check.judge.generator is explicit
    assert check._judge is check.judge


def test_judge_assignment_replaces_legacy_generator_judge():
    check = LLMJudge(
        prompt="Evaluate the answer.",
        generator=Generator(model="openai/gpt-4o-mini"),
    )

    check.judge = BaseJudge.parse("typesafe/jev")

    assert isinstance(check.judge, SOMJudge)
    assert check.generator is None


def test_generator_assignment_replaces_existing_generator():
    check = LLMJudge(
        prompt="Evaluate the answer.",
        generator=Generator(model="openai/gpt-4o-mini"),
    )
    replacement = Generator(model="azure_ai/gpt-5.6-luna")

    check.generator = replacement

    assert check.generator is replacement
    assert isinstance(check.judge, LLMChatJudge)
    assert check.judge.generator is replacement


def test_generator_assignment_replaces_som_judge():
    check = LLMJudge(
        prompt="Evaluate the answer.", judge=BaseJudge.parse("typesafe/jev")
    )
    replacement = Generator(model="openai/gpt-4o-mini")

    check.generator = replacement

    assert check.generator is replacement
    assert isinstance(check.judge, LLMChatJudge)


def test_clearing_generator_clears_explicit_judge():
    check = LLMJudge(
        prompt="Evaluate the answer.",
        generator=Generator(model="openai/gpt-4o-mini"),
    )

    check.generator = None

    assert check.generator is None
    assert check.judge is None


def test_clearing_judge_clears_legacy_generator_view():
    check = LLMJudge(
        prompt="Evaluate the answer.",
        generator=Generator(model="openai/gpt-4o-mini"),
    )

    check.judge = None

    assert check.generator is None
    assert check.judge is None


def test_model_copy_generator_remigrates_to_judge():
    check = LLMJudge(prompt="Evaluate the answer.")
    explicit = Generator(model="openai/gpt-4o-mini")

    copied = check.model_copy(update={"generator": explicit})

    assert copied.generator is explicit
    assert isinstance(copied.judge, LLMChatJudge)
    assert copied.judge.generator is explicit
    assert copied._judge is copied.judge


def test_model_copy_generator_replaces_existing_judge():
    check = LLMJudge(
        prompt="Evaluate the answer.",
        judge=BaseJudge.parse("typesafe/jev"),
    )

    explicit = Generator(model="openai/gpt-4o-mini")
    copied = check.model_copy(update={"generator": explicit})

    assert copied.generator is explicit
    assert isinstance(copied.judge, LLMChatJudge)
    assert copied.judge.generator is explicit


def test_model_copy_generator_none_clears_explicit_judge():
    check = LLMJudge(
        prompt="Evaluate the answer.",
        generator=Generator(model="openai/gpt-4o-mini"),
    )

    copied = check.model_copy(update={"generator": None})

    assert copied.generator is None
    assert copied.judge is None


def test_model_copy_judge_none_clears_legacy_generator_view():
    check = LLMJudge(
        prompt="Evaluate the answer.",
        generator=Generator(model="openai/gpt-4o-mini"),
    )

    copied = check.model_copy(update={"judge": None})

    assert copied.generator is None
    assert copied.judge is None


def test_model_copy_parses_loose_judge_update():
    copied = LLMJudge(prompt="Evaluate the answer.").model_copy(
        update={"judge": "typesafe/jev"}
    )

    assert isinstance(copied.judge, BaseJudge)
    assert copied.judge.kind == "som"


def test_model_copy_rejects_explicit_generator_and_judge():
    check = LLMJudge(prompt="Evaluate the answer.")

    with pytest.raises(ValueError, match="both 'generator' and 'judge'"):
        check.model_copy(
            update={
                "generator": Generator(model="openai/gpt-4o-mini"),
                "judge": "typesafe/jev",
            }
        )


def test_model_copy_rejects_generator_none_with_judge():
    check = LLMJudge(prompt="Evaluate the answer.")

    with pytest.raises(ValueError, match="both 'generator' and 'judge'"):
        check.model_copy(update={"generator": None, "judge": "typesafe/jev"})


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
