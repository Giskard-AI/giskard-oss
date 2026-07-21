"""Split-question knowledge-base quality scenario generator."""

from typing import Any, ClassVar, override

import numpy as np
from giskard.checks import Scenario, Trace
from pydantic import Field

from ..base import ScenarioContext, TargetMode
from .base import KnowledgeBaseScenarioGenerator

SPLIT_QUESTIONS_QUALITY_TAGS = ["quality:split-questions", "component:history"]


class SplitQuestionsScenarioGenerator(KnowledgeBaseScenarioGenerator):
    """Generate two-message document-grounded questions.

    The first user message supplies context without asking the question. The
    second asks the question in a way that depends on the first message.
    """

    scenario_name_prefix: ClassVar[str] = "Knowledge Base Split Questions"
    prompt_path: ClassVar[str] = (
        "giskard.scan::scenarios/knowledge_base_split_question.j2"
    )
    quality_tags: ClassVar[list[str]] = SPLIT_QUESTIONS_QUALITY_TAGS

    max_turns: int = Field(default=2, ge=2, le=2)

    @override
    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
        target_mode: TargetMode = "multiturn",
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        if self._skip_for_singleturn(target_mode):
            return []
        return await super().generate_scenario(
            context,
            max_scenarios=max_scenarios,
            rng=rng,
            target_mode=target_mode,
        )
