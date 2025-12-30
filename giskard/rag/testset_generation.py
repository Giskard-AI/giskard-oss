from typing import Optional, Sequence, Union

import itertools
import logging
from pathlib import Path

from ..utils.analytics_collector import analytics
from .knowledge_base import KnowledgeBase
from .question_generators import (
    QuestionGenerator,
    complex_questions,
    conversational_questions,
    distracting_questions,
    double_questions,
    simple_questions,
    situational_questions,
)
from .question_generators.utils import maybe_tqdm
from .testset import QATestset

logger = logging.getLogger(__name__)


def generate_testset(
    knowledge_base: KnowledgeBase,
    num_questions: Optional[int] = 120,
    question_generators: Optional[Union[QuestionGenerator, Sequence[QuestionGenerator]]] = None,
    language: Optional[str] = "en",
    agent_description: Optional[str] = "This agent is a chatbot that answers question from users.",
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 10,
) -> QATestset:
    """Generate a testset from a knowledge base.

    Parameters
    ----------
    knowledge_base : KnowledgeBase
        The knowledge base to generate questions from.
    num_questions : int
        The number of questions to generate. By default 120.
    question_generators : Union[BaseQuestionModifier, Sequence[BaseQuestionModifier]]
        Question generators to use for question generation. If multiple generator are specified,
        `num_questions` will be generated with each generators. If not specified, all available question generators will be used.
    language : str, optional
        The language to use for question generation. The default is "en" to generate questions in english.
    agent_description : str, optional
        Description of the agent to be evaluated. This will be used in the prompt for question generation to get more fitting questions.
    checkpoint_path : str, optional
        Path to save intermediate results during generation. If provided, progress will be saved
        periodically and can be resumed if generation fails. If the file exists, generation will
        resume from the checkpoint. Default is None (no checkpointing).
    checkpoint_every : int, optional
        Number of questions to generate before saving a checkpoint. Default is 10.
        Only used when checkpoint_path is provided.

    Returns
    -------
    QATestset
        The generated test set.
    """

    if question_generators is None:
        question_generators = [
            simple_questions,
            complex_questions,
            distracting_questions,
            situational_questions,
            double_questions,
            conversational_questions,
        ]

    if not isinstance(question_generators, Sequence):
        question_generators = [question_generators]

    # Ensure topics are computed and documents populated (@TODO: remove this)
    _ = knowledge_base.topics

    # Load existing checkpoint if available
    questions = []
    num_existing = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            existing_testset = QATestset.load(checkpoint_path)
            questions = list(existing_testset.samples)
            num_existing = len(questions)
            logger.info(f"Resuming from checkpoint: {num_existing} questions already generated")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}. Starting fresh.")
            questions = []
            num_existing = 0

    # Calculate remaining questions to generate
    num_remaining = max(0, num_questions - num_existing)
    if num_remaining == 0:
        logger.info(f"Checkpoint already contains {num_existing} questions, nothing to generate")
        return QATestset(questions)

    # Generate questions
    # @TODO: fix this ugly way to distribute the questions across generators
    generator_num_questions = [
        num_remaining // len(question_generators) + (1 if i < num_remaining % len(question_generators) else 0)
        for i in range(len(question_generators))
    ]

    main_generator = itertools.chain.from_iterable(
        [
            generator.generate_questions(
                knowledge_base,
                num_questions=n,
                agent_description=agent_description,
                language=language,
            )
            for generator, n in zip(question_generators, generator_num_questions)
        ]
    )

    # Generate questions with checkpointing
    generated_since_checkpoint = 0
    for question in maybe_tqdm(main_generator, total=num_remaining, desc="Generating questions"):
        # Add topic metadata
        topic_id = knowledge_base[question.metadata["seed_document_id"]].topic_id
        question.metadata["topic"] = knowledge_base.topics[topic_id]

        questions.append(question)
        generated_since_checkpoint += 1

        # Save checkpoint periodically
        if checkpoint_path and generated_since_checkpoint >= checkpoint_every:
            try:
                QATestset(questions).save(checkpoint_path)
                logger.debug(f"Checkpoint saved: {len(questions)} questions")
                generated_since_checkpoint = 0
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")

    # Save final checkpoint
    if checkpoint_path and generated_since_checkpoint > 0:
        try:
            QATestset(questions).save(checkpoint_path)
            logger.info(f"Final checkpoint saved: {len(questions)} questions")
        except Exception as e:
            logger.warning(f"Failed to save final checkpoint: {e}")

    analytics.track(
        "raget:testset-generation",
        {
            "num_questions": num_questions,
            "language": language,
            "agent_description": agent_description,
            "question_generators": [qg.__class__.__name__ for qg in question_generators],
            "knowledge_base_size": len(knowledge_base._documents),
            "resumed_from_checkpoint": num_existing > 0,
            "num_resumed": num_existing,
        },
    )
    return QATestset(questions)
