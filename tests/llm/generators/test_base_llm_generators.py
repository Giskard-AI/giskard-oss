import json
from unittest.mock import Mock

import pytest

from giskard.datasets.base import Dataset
from giskard.llm.client import ChatMessage
from giskard.llm.errors import LLMGenerationError
from giskard.llm.generators.adversarial import AdversarialDataGenerator
from giskard.llm.generators.base import LLMBasedDataGenerator
from giskard.llm.generators.implausible import ImplausibleDataGenerator
from giskard.llm.generators.sycophancy import SycophancyDataGenerator


@pytest.mark.parametrize(
    "Generator,args,kwargs",
    [
        (
            LLMBasedDataGenerator,
            [],
            {
                "prompt": "You need to generate data for this model: {model.description} with features {model.feature_names}"
            },
        ),
        (ImplausibleDataGenerator, [], {}),
        (AdversarialDataGenerator, ["demo", "demo"], {}),
    ],
)
def test_generator_returns_dataset(Generator, args, kwargs):
    llm_client = Mock()
    llm_client.complete.side_effect = [
        ChatMessage(
            role="assistant",
            content=json.dumps(
                {
                    "inputs": [
                        {"question": "What is the meaning of life?", "other_feature": "test"},
                        {
                            "question": "What is the airspeed velocity of an unladen swallow?",
                            "other_feature": "pass",
                        },
                    ]
                }
            ),
        )
    ]

    model = Mock()
    model.feature_names = ["question", "other_feature"]
    model.name = "Mock model for test"
    model.description = "This is a model for testing purposes"

    generator = Generator(
        *args,
        **kwargs,
        llm_client=llm_client,
        llm_temperature=1.416,
    )

    dataset = generator.generate_dataset(model, num_samples=2)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset.df.iloc[0]["question"] == "What is the meaning of life?"
    assert dataset.df.iloc[1]["other_feature"] == "pass"

    llm_client.complete.assert_called_once()

    called_prompt = llm_client.complete.call_args[1]["messages"][-1].content
    called_temperature = llm_client.complete.call_args[1]["temperature"]

    assert model.description in called_prompt
    assert "This is a model for testing purposes" in called_prompt
    assert "other_feature" in called_prompt
    assert called_temperature == 1.416


@pytest.mark.parametrize(
    "Generator,args,kwargs",
    [
        (LLMBasedDataGenerator, [], {"prompt": "Dummy prompt for {model.name}"}),
        (ImplausibleDataGenerator, [], {}),
        (SycophancyDataGenerator, [], {}),
        (AdversarialDataGenerator, ["demo", "demo"], {}),
    ],
)
def test_generator_raises_generation_error_if_parsing_fails(Generator, args, kwargs):
    # Missing tool call
    llm_client = Mock()
    llm_client.complete.side_effect = [ChatMessage("assistant", "Sorry, I can't.")]

    model = Mock()
    model.feature_names = ["question", "other_feature"]
    model.name = "Mock model for test"
    model.description = "This is a model for testing purposes"

    generator = Generator(*args, **kwargs, llm_client=llm_client)

    with pytest.raises(LLMGenerationError):
        generator.generate_dataset(model, num_samples=2)

    # Wrong tool call
    llm_client = Mock()
    llm_client.complete.side_effect = [
        ChatMessage(
            role="assistant",
            content=json.dumps({"wrong_key": "wrong_value"}),
        )
    ]

    model = Mock()
    model.feature_names = ["question", "other_feature"]
    model.name = "Mock model for test"
    model.description = "This is a model for testing purposes"

    generator = Generator(*args, **kwargs, llm_client=llm_client)

    with pytest.raises(LLMGenerationError):
        generator.generate_dataset(model, num_samples=2)


@pytest.mark.parametrize(
    "Generator,args,kwargs",
    [
        (LLMBasedDataGenerator, [], {"prompt": "Dummy prompt for {model.name}"}),
        (ImplausibleDataGenerator, [], {}),
    ],
)
def test_generator_casts_based_on_column_types(Generator, args, kwargs):
    llm_client = Mock()
    llm_client.complete.side_effect = [
        ChatMessage(
            role="assistant",
            content=json.dumps(
                {
                    "inputs": [
                        {"question": True, "other_feature": 1},
                        {"question": False, "other_feature": 2},
                    ]
                },
            ),
        )
    ] * 2

    model = Mock()
    model.feature_names = ["question", "other_feature"]
    model.name = "Mock model for test"
    model.description = "This is a model for testing purposes"

    generator = Generator(*args, **kwargs, llm_client=llm_client)

    dataset = generator.generate_dataset(model, num_samples=2)

    assert dataset.column_types["question"] == "numeric"
    assert dataset.column_types["other_feature"] == "numeric"

    dataset = generator.generate_dataset(
        model, num_samples=2, column_types={"question": "text", "other_feature": "numeric"}
    )

    assert dataset.column_types["question"] == "text"
    assert dataset.column_types["other_feature"] == "numeric"


@pytest.mark.parametrize(
    "Generator,args,kwargs",
    [
        (LLMBasedDataGenerator, [], {"prompt": "Dummy prompt for {model.name}\n### LANGUAGES\n{languages}"}),
        (ImplausibleDataGenerator, [], {}),
        (AdversarialDataGenerator, ["demo", "demo"], {}),
    ],
)
def test_generator_adds_languages_requirements_in_prompts(Generator, args, kwargs):
    llm_client = Mock()
    llm_client.complete.side_effect = [
        ChatMessage(
            role="assistant",
            content=json.dumps(
                {
                    "inputs": [
                        {"question": "What is the meaning of life?", "other_feature": "test"},
                        {
                            "question": "Quel est le rôle des gaz à effet de serre dans le réchauffement climatique??",
                            "other_feature": "pass",
                        },
                    ]
                }
            ),
        )
    ]

    model = Mock()
    model.feature_names = ["question", "other_feature"]
    model.name = "Mock model for test"
    model.description = "This is a model for testing purposes"

    generator = Generator(
        *args,
        **kwargs,
        llm_client=llm_client,
        llm_temperature=1.416,
        languages=["en", "fr"],
    )

    dataset = generator.generate_dataset(model, num_samples=2)

    llm_client.complete.assert_called_once()

    called_prompt = llm_client.complete.call_args[1]["messages"][-1].content

    assert isinstance(dataset, Dataset)
    assert "### LANGUAGES\nen, fr" in called_prompt


@pytest.mark.parametrize(
    "Generator,args,kwargs",
    [
        (LLMBasedDataGenerator, [], {"prompt": "Dummy prompt for {model.name}\n### LANGUAGES\n{languages}"}),
        (ImplausibleDataGenerator, [], {}),
        (AdversarialDataGenerator, ["demo", "demo"], {}),
    ],
)
def test_generator_empty_languages_requirements(Generator, args, kwargs):
    llm_client = Mock()
    llm_client.complete.side_effect = [
        ChatMessage(
            role="assistant",
            content=json.dumps(
                {
                    "inputs": [
                        {"question": "What is the meaning of life?", "other_feature": "test"},
                        {
                            "question": "Quel est le rôle des gaz à effet de serre dans le réchauffement climatique??",
                            "other_feature": "pass",
                        },
                    ]
                }
            ),
        )
    ]

    model = Mock()
    model.feature_names = ["question", "other_feature"]
    model.name = "Mock model for test"
    model.description = "This is a model for testing purposes"

    generator = Generator(
        *args,
        **kwargs,
        llm_client=llm_client,
        llm_temperature=1.416,
        languages=[],
    )

    dataset = generator.generate_dataset(model, num_samples=2)

    llm_client.complete.assert_called_once()

    called_prompt = llm_client.complete.call_args[1]["messages"][-1].content

    assert isinstance(dataset, Dataset)
    assert "### LANGUAGES\nen" in called_prompt


@pytest.mark.parametrize(
    "Generator,args,kwargs",
    [
        (LLMBasedDataGenerator, [], {"prompt": "Dummy prompt for {model.name}"}),
        (ImplausibleDataGenerator, [], {}),
        (AdversarialDataGenerator, ["demo", "demo"], {}),
    ],
)
def test_generator_aligns_column_names_to_feature_names(Generator, args, kwargs):
    """Test that generator renames columns to match model's feature_names.

    This tests the fix for issue #2213 where different detectors would generate
    datasets with inconsistent column names (e.g., 'user_input', 'query', 'user_question')
    instead of the model's expected feature_names.
    """
    llm_client = Mock()
    # LLM generates columns with different names than feature_names
    llm_client.complete.side_effect = [
        ChatMessage(
            role="assistant",
            content=json.dumps(
                {
                    "inputs": [
                        {"user_input": "Hello, world!", "user_name": "Alice"},
                        {"user_input": "How are you?", "user_name": "Bob"},
                    ]
                }
            ),
        )
    ]

    model = Mock()
    # Model expects different column names
    model.feature_names = ["message", "sender"]
    model.name = "Mock model for test"
    model.description = "This is a model for testing purposes"

    generator = Generator(*args, **kwargs, llm_client=llm_client)

    dataset = generator.generate_dataset(model, num_samples=2)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    # Columns should be renamed to match feature_names
    assert list(dataset.df.columns) == ["message", "sender"]
    assert dataset.df.iloc[0]["message"] == "Hello, world!"
    assert dataset.df.iloc[1]["sender"] == "Bob"


@pytest.mark.parametrize(
    "Generator,args,kwargs",
    [
        (LLMBasedDataGenerator, [], {"prompt": "Dummy prompt for {model.name}"}),
        (ImplausibleDataGenerator, [], {}),
    ],
)
def test_generator_preserves_matching_column_names(Generator, args, kwargs):
    """Test that generator preserves columns when they already match feature_names."""
    llm_client = Mock()
    llm_client.complete.side_effect = [
        ChatMessage(
            role="assistant",
            content=json.dumps(
                {
                    "inputs": [
                        {"question": "What is AI?", "context": "tech"},
                        {"question": "How does ML work?", "context": "science"},
                    ]
                }
            ),
        )
    ]

    model = Mock()
    model.feature_names = ["question", "context"]
    model.name = "Mock model for test"
    model.description = "This is a model for testing purposes"

    generator = Generator(*args, **kwargs, llm_client=llm_client)

    dataset = generator.generate_dataset(model, num_samples=2)

    assert isinstance(dataset, Dataset)
    assert list(dataset.df.columns) == ["question", "context"]
    assert dataset.df.iloc[0]["question"] == "What is AI?"
