from collections.abc import Sequence
from typing import Literal, override
from unittest.mock import MagicMock

import giskard.agents.som as som_module
import giskard.checks.settings as settings_module
import pytest
from giskard.agents import (
    BaseEmbeddingModel,
    BaseGenerator,
    BaseSOM,
    EmbeddingModel,
    Generator,
    SOMResponse,
)
from giskard.checks import (
    SOMJudgeGenerator,
    get_default_embedding_model,
    get_default_judge,
    set_default_embedding_model,
    set_default_judge,
)
from giskard.checks.settings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    get_default_generator,
    get_settings,
    set_default_generator,
)
from giskard.llm.types import ChatMessage


@BaseSOM.register("checks_settings_test_som")
class CustomSOM(BaseSOM):
    @override
    async def predict(
        self,
        messages: Sequence[ChatMessage],
        question: str,
        *,
        timeout: float | int | None = None,
    ) -> SOMResponse:
        return SOMResponse(model=self.model, probability=0.9)


def test_default_generator_uses_settings_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_DEFAULT_MODEL", "google/gemini-3.5-flash")

    generator = get_default_generator()

    assert isinstance(generator, Generator)
    assert generator.model == "google/gemini-3.5-flash"


def test_default_generator_falls_back_to_builtin_default():
    settings_module._default_generator = None

    generator = get_default_generator()

    assert isinstance(generator, Generator)
    assert generator.model == DEFAULT_MODEL


def test_set_default_generator_overrides_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_DEFAULT_MODEL", "google/gemini-3.5-flash")
    explicit = Generator(model="anthropic/claude-haiku-4-5-20251001")

    set_default_generator(explicit)

    assert get_default_generator() is explicit


def test_set_default_generator_accepts_model_string(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_DEFAULT_MODEL", "google/gemini-3.5-flash")

    set_default_generator("azure/gpt-5.6-luna")

    generator = get_default_generator()
    assert isinstance(generator, Generator)
    assert generator.model == "azure/gpt-5.6-luna"


def test_default_judge_falls_back_to_default_generator():
    generator = Generator(model="azure_ai/gpt-5.6-luna")
    set_default_generator(generator)

    assert get_default_judge() is generator


def test_default_judge_falls_back_to_generator_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GISKARD_CHECKS_DEFAULT_MODEL", "google/gemini-3.5-flash")

    judge = get_default_judge()

    assert isinstance(judge, Generator)
    assert judge.model == "google/gemini-3.5-flash"


def test_set_default_judge_preserves_custom_backend():
    judge = MagicMock(spec=BaseGenerator)

    set_default_judge(judge)

    assert get_default_judge() is judge


@pytest.mark.parametrize("model_type", [None, "llm"])
def test_set_default_judge_accepts_llm_model_string(
    model_type: Literal["llm", "som"] | None,
):
    set_default_judge("openai/gpt-4o-mini", model_type=model_type)

    judge = get_default_judge()

    assert isinstance(judge, Generator)
    assert judge.model == "openai/gpt-4o-mini"


@pytest.mark.parametrize("model_type", [None, "som"])
@pytest.mark.parametrize(
    ("model", "native_model"),
    [("typesafe/jev", "jev-latest"), ("typesafe/jev-preview", "jev-preview")],
)
def test_set_default_judge_accepts_typesafe_model_string(
    model: str, native_model: str, model_type: Literal["llm", "som"] | None
):
    set_default_judge(model, model_type=model_type)

    judge = get_default_judge()

    assert isinstance(judge, SOMJudgeGenerator)
    assert judge.model.model == native_model


def test_set_default_judge_accepts_custom_som_model():
    model = CustomSOM(model="example-v1")

    set_default_judge(model)

    judge = get_default_judge()
    assert isinstance(judge, SOMJudgeGenerator)
    assert judge.model is model


@pytest.mark.parametrize("model_type", [None, "som"])
def test_set_default_judge_resolves_another_registered_som_provider(
    monkeypatch: pytest.MonkeyPatch, model_type: Literal["llm", "som"] | None
):
    monkeypatch.setitem(som_module._PROVIDERS, "example", CustomSOM)

    set_default_judge("example/new-som", model_type=model_type)

    judge = get_default_judge()
    assert isinstance(judge, SOMJudgeGenerator)
    assert isinstance(judge.model, CustomSOM)
    assert judge.model.model == "new-som"


def test_default_judge_and_generator_are_independent():
    generation = Generator(model="azure_ai/gpt-5.6-luna")
    judge = Generator(model="openai/gpt-4o-mini")
    set_default_generator(generation)
    set_default_judge(judge)

    assert get_default_generator() is generation
    assert get_default_judge() is judge

    updated_generation = Generator(model="google/gemini-3.5-flash")
    set_default_generator(updated_generation)

    assert get_default_generator() is updated_generation
    assert get_default_judge() is judge


def test_reset_default_judge_restores_generator_fallback():
    generator = Generator(model="azure_ai/gpt-5.6-luna")
    set_default_generator(generator)
    set_default_judge("typesafe/jev")

    set_default_judge(None)

    assert get_default_judge() is generator


@pytest.mark.parametrize(
    ("model", "model_type", "message"),
    [
        ("typesafe/jev", "llm", "Use model_type="),
        ("openai/gpt-4o-mini", "som", "Use model_type="),
        ("jev", "som", "Use model_type="),
        ("typesafe/", "som", "Specify a SOM model as 'provider/model'"),
        ("openai/gpt-4o-mini", "unknown", "Use model_type="),
    ],
)
def test_invalid_judge_configuration_preserves_current_default(
    model: str, model_type: Literal["llm", "som"], message: str
):
    original = Generator(model="openai/gpt-4o-mini")
    set_default_judge(original)

    with pytest.raises(ValueError, match=message):
        set_default_judge(model, model_type=model_type)

    assert get_default_judge() is original


@pytest.mark.parametrize(
    "judge",
    [None, Generator(model="openai/gpt-4o-mini"), CustomSOM(model="example-v1")],
)
def test_set_default_judge_rejects_model_type_without_model_string(
    judge: BaseGenerator | BaseSOM | None,
):
    with pytest.raises(
        ValueError, match="only supported with a model identifier string"
    ):
        set_default_judge(judge, model_type="llm")


def test_default_embedding_model_uses_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "GISKARD_CHECKS_DEFAULT_EMBEDDING_MODEL", "google/gemini-embedding-001"
    )

    embedding_model = get_default_embedding_model()

    assert isinstance(embedding_model, EmbeddingModel)
    assert embedding_model.model == "google/gemini-embedding-001"


def test_default_embedding_model_falls_back_to_builtin_default():
    embedding_model = get_default_embedding_model()

    assert isinstance(embedding_model, EmbeddingModel)
    assert embedding_model.model == DEFAULT_EMBEDDING_MODEL


def test_set_default_embedding_model_accepts_model_string(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "GISKARD_CHECKS_DEFAULT_EMBEDDING_MODEL", "google/gemini-embedding-001"
    )

    set_default_embedding_model("text-embedding-3-large")

    embedding = get_default_embedding_model()
    assert isinstance(embedding, EmbeddingModel)
    assert embedding.model == "text-embedding-3-large"


def test_set_default_embedding_model_preserves_custom_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "GISKARD_CHECKS_DEFAULT_EMBEDDING_MODEL", "google/gemini-embedding-001"
    )
    embedding = MagicMock(spec=BaseEmbeddingModel)

    set_default_embedding_model(embedding)

    assert get_default_embedding_model() is embedding


def test_reset_default_embedding_model_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "GISKARD_CHECKS_DEFAULT_EMBEDDING_MODEL", "google/gemini-embedding-001"
    )
    set_default_embedding_model("text-embedding-3-large")

    set_default_embedding_model(None)

    embedding = get_default_embedding_model()
    assert isinstance(embedding, EmbeddingModel)
    assert embedding.model == "google/gemini-embedding-001"


def test_reset_default_embedding_model_restores_builtin_default():
    set_default_embedding_model("text-embedding-3-large")

    set_default_embedding_model(None)

    embedding = get_default_embedding_model()
    assert isinstance(embedding, EmbeddingModel)
    assert embedding.model == DEFAULT_EMBEDDING_MODEL


def test_default_embedding_is_independent_of_generator_and_judge():
    generator = Generator(model="azure_ai/gpt-5.6-luna")
    judge = Generator(model="openai/gpt-4o-mini")
    embedding = MagicMock(spec=BaseEmbeddingModel)
    set_default_generator(generator)
    set_default_judge(judge)

    set_default_embedding_model(embedding)

    assert get_default_generator() is generator
    assert get_default_judge() is judge
    assert get_default_embedding_model() is embedding

    set_default_generator("google/gemini-3.5-flash")
    set_default_judge("typesafe/jev")

    assert get_default_embedding_model() is embedding


def test_settings_max_reported_failures_validation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_MAX_REPORTED_FAILURES", "3")
    assert get_settings().max_reported_failures == 3

    monkeypatch.setenv("GISKARD_CHECKS_MAX_REPORTED_FAILURES", "invalid")
    assert get_settings().max_reported_failures is None

    monkeypatch.setenv("GISKARD_CHECKS_MAX_REPORTED_FAILURES", "-1")
    assert get_settings().max_reported_failures is None

    monkeypatch.setenv("GISKARD_CHECKS_MAX_REPORTED_FAILURES", "true")
    assert get_settings().max_reported_failures is None


def test_settings_disable_rich_pretty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_DISABLE_RICH_PRETTY", "true")
    assert get_settings().disable_rich_pretty is True
