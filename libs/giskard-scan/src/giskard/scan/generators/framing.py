"""Scenario generators for framing injection attacks.

Framing attacks hide a harmful request inside an innocuous-looking scenario
(a deceased-grandparent roleplay, a symbolic math problem, a scholarly
citation, an academic Likert rubric) so that a target LLM is more likely to
comply with the hidden request than it would with the raw prompt.

Like :class:`~giskard.scan.generators.gcg.GCGInjectionScenarioGenerator`, these
generators reuse the HarmBench harmful-prompt dataset as a payload library. They
subclass :class:`HuggingFaceDatasetScenarioGenerator` to inherit the dataset's
loading, then transform each loaded scenario: the raw dataset prompt is replaced
by an :class:`~giskard.checks.LLMGenerator` that wraps the harmful *behavior*
(carried in the scenario's annotations) in the framing scenario at run time. The
dataset's own HarmBench safety judge is preserved, since a successful framing
attack is still scored as compliance with the hidden behavior.
"""

from typing import Any, ClassVar, override

import numpy as np
from giskard.checks import (
    DatasetInputGenerator,
    Interact,
    LLMGenerator,
    Scenario,
    Trace,
)
from pydantic import Field

from .base import ScenarioContext, TargetMode
from .huggingface import HuggingFaceDatasetScenarioGenerator

# The HarmBench behaviors are sourced from the English subset; the framing
# LLMGenerator then translates the framed wrapper into the agent's languages at
# run time (the requested languages are carried in the scenario's annotations).
_BEHAVIOR_LANGUAGE = ["en"]

# HarmBench functional categories that framing does not target: copyright
# (reproducing protected text) and contextual (needs extra context the framing
# wrapper cannot carry). Only "standard" harmful behaviors are framed.
_EXCLUDED_FUNCTIONAL_CATEGORIES = frozenset({"copyright", "contextual"})

_FRAMING_TAGS = [
    "threat-type:prompt-injection",
    "owasp:llm-top-10-2025:LLM01",
]


class FramingScenarioGenerator(HuggingFaceDatasetScenarioGenerator):
    """Base generator that frames HarmBench behaviors inside a scenario.

    Loads the HarmBench ``en`` subset, keeps only ``standard`` behaviors, and
    rebuilds each scenario so the harmful prompt is produced at run time by an
    :class:`~giskard.checks.LLMGenerator` rendering :attr:`prompt_path`. The
    behavior remains available to that template (and to the preserved safety
    judge) via ``trace.annotations.behavior``.

    Subclasses set :attr:`prompt_path`, :attr:`name_prefix`, and
    :attr:`max_turns`.

    Attributes:
        prompt_path: ``giskard.scan::`` template reference for the framing
            generator. Subclasses must override.
        name_prefix: Human-readable prefix for generated scenario names.
        max_turns: Conversation turns the framing generator may take. Capped to
            ``1`` automatically when ``target_mode="singleturn"``.
        repo_id: HarmBench dataset of harmful base behaviors.
    """

    prompt_path: ClassVar[str] = ""
    name_prefix: ClassVar[str] = "Framing"
    max_turns: ClassVar[int] = 1

    repo_id: str = Field(default="giskardai/harmbench-scenarios")
    repo_allow_commercial_use: bool = Field(default=True)

    @override
    def load_scenarios(
        self, description: str, languages: list[str]
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        # Source the behavior from the English subset, but record the agent's
        # actual languages so the framing generator translates the wrapper.
        base_scenarios = super().load_scenarios(description, _BEHAVIOR_LANGUAGE)
        return [
            self._frame_scenario(scenario, languages)
            for scenario in base_scenarios
            if self._is_framable(scenario)
        ]

    @staticmethod
    def _is_framable(scenario: Scenario[Any, Any, Trace[Any, Any]]) -> bool:
        """Whether ``scenario``'s HarmBench behavior is a framing target.

        Only ``standard`` behaviors are framed; ``copyright`` and ``contextual``
        behaviors are excluded. Behaviors with no recorded functional category
        are kept (treated as framable).
        """
        category = scenario.annotations.get("functional_category")
        return category not in _EXCLUDED_FUNCTIONAL_CATEGORIES

    def _frame_scenario(
        self,
        scenario: Scenario[Any, Any, Trace[Any, Any]],
        languages: list[str],
    ) -> Scenario[Any, Any, Trace[Any, Any]]:
        """Return a copy of ``scenario`` whose prompt is produced by framing.

        The dataset-backed prompt interaction is replaced with an
        :class:`~giskard.checks.LLMGenerator` rendering :attr:`prompt_path`; the
        scenario's checks (the HarmBench safety judge) and the ``behavior``
        annotation are preserved. The ``languages`` annotation is set to the
        agent's requested languages (not the English behavior subset) so the
        framing generator translates the wrapper. The copy is renamed and tagged
        for grouping.
        """
        new_steps = [
            step.model_copy(
                update={
                    "interacts": [
                        self._frame_interact(interact) for interact in step.interacts
                    ]
                }
            )
            for step in scenario.steps
        ]
        return scenario.model_copy(
            update={
                "name": f"{self.name_prefix} - {scenario.name}",
                "steps": new_steps,
                "annotations": {**scenario.annotations, "languages": languages},
                # Append, don't replace: the dataset's own tags (threat-type,
                # dataset:harmbench, semantic-category, ...) must survive.
                "tags": [*scenario.tags, *_FRAMING_TAGS],
            }
        )

    def _frame_interact(self, interact: Any) -> Interact[Any, Any, Any]:
        """Replace a dataset-backed interaction's prompt with the framing generator."""
        if not isinstance(interact, Interact):
            raise TypeError(f"Expected an Interact spec, got {type(interact).__name__}")
        if not isinstance(interact.inputs, DatasetInputGenerator):
            raise TypeError(
                "Framing injection expects dataset-backed prompts "
                f"(DatasetInputGenerator), got {type(interact.inputs).__name__}"
            )
        return interact.model_copy(
            update={
                "inputs": LLMGenerator(
                    prompt_path=self.prompt_path, max_steps=self.max_turns
                )
            }
        )


class GrandmaFramingScenarioGenerator(FramingScenarioGenerator):
    """Frame harmful behaviors as a deceased-grandparent bedtime roleplay.

    Tests whether emotional roleplay framing ("act as my late grandmother who
    used to tell me ...") elicits compliance with a hidden harmful request.
    """

    prompt_path: ClassVar[str] = "giskard.scan::scenarios/grandma_framing.j2"
    name_prefix: ClassVar[str] = "Grandma Framing"
    max_turns: ClassVar[int] = 1


class MathFramingScenarioGenerator(FramingScenarioGenerator):
    """Frame harmful behaviors as a symbolic mathematics problem.

    Tests whether casting a harmful request as set-theory/abstract-algebra
    notation plus a "real-world example" request bypasses safety measures.

    References:
        https://arxiv.org/pdf/2409.11445
    """

    prompt_path: ClassVar[str] = "giskard.scan::scenarios/math_framing.j2"
    name_prefix: ClassVar[str] = "Math Framing"
    max_turns: ClassVar[int] = 1


class CitationFramingScenarioGenerator(FramingScenarioGenerator):
    """Frame harmful behaviors as a request to summarize a scholarly citation.

    Tests whether asking the target to summarize a fabricated but
    realistic-looking citation (DarkCite-style) elicits the hidden content.

    References:
        DarkCite: https://arxiv.org/pdf/2411.11407
    """

    prompt_path: ClassVar[str] = "giskard.scan::scenarios/citation_framing.j2"
    name_prefix: ClassVar[str] = "Citation Framing"
    max_turns: ClassVar[int] = 1


class LikertFramingScenarioGenerator(FramingScenarioGenerator):
    """Frame harmful behaviors as a two-turn academic Likert-rubric exercise.

    Turn 1 sets up a content-moderation rubric; turn 2 requests a high-score
    example, which surfaces the hidden behavior. This attack is multi-turn by
    design and is skipped entirely for single-turn targets.

    References:
        https://unit42.paloaltonetworks.com/multi-turn-technique-jailbreaks
    """

    prompt_path: ClassVar[str] = "giskard.scan::scenarios/likert_framing.j2"
    name_prefix: ClassVar[str] = "Likert Framing"
    max_turns: ClassVar[int] = 2

    @override
    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
        target_mode: TargetMode = "multiturn",
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        # The Likert attack needs its setup turn before the injection turn; it
        # has no meaningful single-turn form, so skip rather than cap to 1.
        if self._skip_for_singleturn(target_mode):
            return []
        return await super().generate_scenario(context, max_scenarios, rng, target_mode)
