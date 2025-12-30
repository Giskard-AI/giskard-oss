import logging
from unittest.mock import Mock, MagicMock

from giskard.llm.client.base import ChatMessage
from giskard.rag.question_generators.simple_questions import SimpleQuestionsGenerator
from giskard.rag.question_generators.base import GenerateFromSingleQuestionMixin, _LLMBasedQuestionGenerator


def test_base_generator():
    llm_client = Mock()
    llm_client.complete = Mock(
        return_value=ChatMessage(
            role="assistant",
            content='{"question": "Where is Camembert from?", "answer": "Camembert was created in Normandy, in the northwest of France."}',
        )
    )

    base_generator = SimpleQuestionsGenerator(llm_client=llm_client)
    completion = base_generator._llm_complete(messages=[])

    assert isinstance(completion, dict)
    assert completion["question"] == "Where is Camembert from?"
    assert completion["answer"] == "Camembert was created in Normandy, in the northwest of France."


def test_json_parsing(caplog):
    llm_client = Mock()
    llm_client.complete.side_effect = [
        ChatMessage(
            role="assistant",
            content='```json {"question": "Where is Camembert from?", "answer": "Camembert was created in Normandy, in the northwest of France."} ```',
        )
    ]

    base_generator = SimpleQuestionsGenerator(llm_client=llm_client)
    with caplog.at_level(logging.DEBUG, logger="giskard.rag"):
        completion = base_generator._llm_complete(messages=[])

    assert isinstance(completion, dict)
    assert completion["question"] == "Where is Camembert from?"
    assert completion["answer"] == "Camembert was created in Normandy, in the northwest of France."

    assert "JSON decoding error" in caplog.text

    llm_client.complete.side_effect = [
        ChatMessage(
            role="assistant",
            content='```json {"question": "Where is Camembert from?", "answer": "Camembert was created in Normandy, in the northwest of France." ```',
        ),
        ChatMessage(
            role="assistant",
            content='{"question": "Where is Camembert from?", "answer": "Camembert was created in Normandy, in the northwest of France."}',
        ),
    ]

    base_generator = SimpleQuestionsGenerator(llm_client=llm_client)
    with caplog.at_level(logging.DEBUG, logger="giskard.rag"):
        completion = base_generator._llm_complete(messages=[])

    assert isinstance(completion, dict)
    assert completion["question"] == "Where is Camembert from?"
    assert completion["answer"] == "Camembert was created in Normandy, in the northwest of France."

    assert "JSON decoding error" in caplog.text


def test_num_retries_default():
    """Test that num_retries defaults to 2."""
    llm_client = Mock()
    generator = SimpleQuestionsGenerator(llm_client=llm_client)
    assert generator._num_retries == 2


def test_num_retries_custom():
    """Test that custom num_retries is respected."""
    llm_client = Mock()
    generator = SimpleQuestionsGenerator(llm_client=llm_client, num_retries=5)
    assert generator._num_retries == 5


def test_num_retries_minimum_is_one():
    """Test that num_retries is at least 1 even if 0 or negative is passed."""
    llm_client = Mock()
    generator = SimpleQuestionsGenerator(llm_client=llm_client, num_retries=0)
    assert generator._num_retries == 1

    generator = SimpleQuestionsGenerator(llm_client=llm_client, num_retries=-1)
    assert generator._num_retries == 1


def test_retry_on_generation_failure(caplog):
    """Test that generation retries on failure and succeeds on retry."""
    llm_client = Mock()
    # First call fails with invalid JSON, second call succeeds
    llm_client.complete.side_effect = [
        ChatMessage(role="assistant", content='invalid json'),
        ChatMessage(role="assistant", content='{"question": "Test?", "answer": "Answer."}'),
        ChatMessage(
            role="assistant",
            content='{"question": "Test?", "answer": "Answer."}',
        ),
    ]

    knowledge_base = MagicMock()
    doc = MagicMock()
    doc.id = "1"
    doc.content = "Test content"
    knowledge_base.get_random_documents.return_value = [doc]
    knowledge_base.get_neighbors.return_value = [doc]

    generator = SimpleQuestionsGenerator(llm_client=llm_client, num_retries=2)

    with caplog.at_level(logging.DEBUG, logger="giskard.rag"):
        questions = list(generator.generate_questions(
            knowledge_base, num_questions=1, agent_description="Test agent", language="en"
        ))

    assert len(questions) == 1
    assert questions[0].question == "Test?"
    assert "Retrying..." in caplog.text


def test_all_retries_fail(caplog):
    """Test that generation skips when all retries fail."""
    llm_client = Mock()
    # All calls fail
    llm_client.complete.side_effect = Exception("LLM error")

    knowledge_base = MagicMock()
    doc = MagicMock()
    doc.id = "1"
    doc.content = "Test content"
    knowledge_base.get_random_documents.return_value = [doc]
    knowledge_base.get_neighbors.return_value = [doc]

    generator = SimpleQuestionsGenerator(llm_client=llm_client, num_retries=3)

    with caplog.at_level(logging.DEBUG, logger="giskard.rag"):
        questions = list(generator.generate_questions(
            knowledge_base, num_questions=1, agent_description="Test agent", language="en"
        ))

    # Should skip the question after all retries fail
    assert len(questions) == 0
    assert "after 3 attempts" in caplog.text
