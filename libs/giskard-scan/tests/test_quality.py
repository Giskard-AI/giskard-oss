from collections.abc import Sequence
from typing import Any, override

import giskard.checks.settings as settings
import giskard.scan.quality as quality_module
import numpy as np
import pytest
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import Equals, Scenario, Trace
from giskard.checks.core.result import SuiteResult
from giskard.checks.scenarios.suite import Suite
from giskard.llm.types import AssistantMessage, ChatMessage, Choice, CompletionResponse
from giskard.scan.generators.base import ScenarioContext, ScenarioGenerator
from giskard.scan.generators.knowledge_base import (
    HallucinationScenarioGenerator,
    MultiTopicScenarioGenerator,
    OutOfScopeScenarioGenerator,
    SplitQuestionsScenarioGenerator,
    SycophancyScenarioGenerator,
)
from giskard.scan.quality import quality_scan, quality_suite_generator_registry
from giskard.scan.utils.knowledge_base import KnowledgeBase
from pydantic import Field


class _DeterministicQualityGenerator(ScenarioGenerator):
    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
        target_mode: str = "multiturn",
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        scenarios = [
            Scenario("passes")
            .interact("accurate")
            .check(Equals(expected_value="accurate", key="trace.last.outputs"))
            .with_tags(["quality:accuracy", "component:retrieval"]),
            Scenario("fails")
            .interact("concise")
            .check(Equals(expected_value="verbose", key="trace.last.outputs"))
            .with_tags(["quality:conciseness", "component:llm"]),
        ]
        return scenarios if max_scenarios is None else scenarios[:max_scenarios]


class _MockRecommendationGenerator(BaseGenerator):
    responses: list[str]
    index: int = 0
    calls: list[Sequence[ChatMessage]] = Field(default_factory=list)

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        _ = params, metadata
        self.calls.append(messages)
        message = AssistantMessage(content=self.responses[self.index])
        self.index += 1
        return CompletionResponse(
            choices=[Choice(message=message, finish_reason="stop", index=0)]
        )


pytestmark = pytest.mark.usefixtures("isolated_quality_registry")


async def test_quality_scan_runs_registry_scenarios_and_returns_suite_result(
    monkeypatch: pytest.MonkeyPatch,
):
    quality_suite_generator_registry.register(_DeterministicQualityGenerator())
    recommendation_generator = _MockRecommendationGenerator(
        responses=["- Improve retrieval grounding."]
    )
    monkeypatch.setattr(settings, "_default_generator", recommendation_generator)
    printed_reports: list[tuple[SuiteResult, str | None]] = []

    def print_report_spy(
        self: SuiteResult,
        console: Any = None,
        group_by: str | None = None,
    ) -> None:
        _ = console
        printed_reports.append((self, group_by))

    monkeypatch.setattr(SuiteResult, "print_report", print_report_spy)

    async def target(inputs: str) -> str:
        return inputs

    result = await quality_scan(
        target=target,
        description="Support agent",
        languages=["en"],
        knowledge_base=["reference document"],
        max_scenarios=2,
        seed=123,
        group_by="quality",
    )

    assert type(result) is SuiteResult
    assert result.recommendation == "- Improve retrieval grounding."
    assert result.passed_count == 1
    assert result.failed_count == 1
    assert result.errored_count == 0
    assert result.skipped_count == 0
    assert result.pass_rate == 0.5
    assert printed_reports == [(result, "quality")]
    assert {scenario.scenario_name for scenario in result.results} == {
        "passes",
        "fails",
    }
    assert {tuple(scenario.tags) for scenario in result.results} == {
        ("quality:accuracy", "component:retrieval"),
        ("quality:conciseness", "component:llm"),
    }

    grouped = result.group_by("quality")
    assert grouped.groups["accuracy"].passed == 1
    assert grouped.groups["accuracy"].failed == 0
    assert grouped.groups["accuracy"].pass_rate == 1.0
    assert grouped.groups["conciseness"].passed == 0
    assert grouped.groups["conciseness"].failed == 1
    assert grouped.groups["conciseness"].pass_rate == 0.0
    rendered_prompt = "\n".join(
        str(message.content) for message in recommendation_generator.calls[0]
    )
    assert "RESULTS BY COMPONENT" in rendered_prompt
    assert "llm: 0/1 passed, 1 failed" in rendered_prompt
    assert "RESULTS BY QUALITY CATEGORY" in rendered_prompt
    assert "conciseness: 0/1 passed, 1 failed" in rendered_prompt
    assert "FAILING SCENARIOS" not in rendered_prompt
    assert "Failure messages" not in rendered_prompt


async def test_quality_scan_forwards_run_options(
    monkeypatch: pytest.MonkeyPatch,
):
    quality_suite_generator_registry.register(_DeterministicQualityGenerator())
    run_kwargs: list[dict[str, object]] = []
    original_run = Suite.run

    async def run_spy(
        self: Suite[Any, Any],
        target: object,
        return_exception: bool = False,
        parallel: bool = False,
        max_concurrency: int | None = None,
        verbose: bool = True,
    ) -> SuiteResult:
        run_kwargs.append(
            {
                "parallel": parallel,
                "max_concurrency": max_concurrency,
                "return_exception": return_exception,
            }
        )
        return await original_run(
            self,
            target,
            return_exception=return_exception,
            parallel=parallel,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )

    monkeypatch.setattr(Suite, "run", run_spy)
    monkeypatch.setattr(SuiteResult, "print_report", lambda self, **_: None)

    async def target(inputs: str) -> str:
        return inputs

    await quality_scan(
        target=target,
        description="Support agent",
        languages=["en"],
        knowledge_base=["reference document"],
        max_scenarios=1,
        parallel=True,
        max_concurrency=3,
        return_exception=True,
    )

    assert run_kwargs == [
        {"parallel": True, "max_concurrency": 3, "return_exception": True}
    ]


async def test_quality_scan_uses_empty_registry_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    printed_reports: list[str | None] = []

    def print_report_spy(
        self: SuiteResult,
        console: Any = None,
        group_by: str | None = None,
    ) -> None:
        _ = self, console
        printed_reports.append(group_by)

    monkeypatch.setattr(SuiteResult, "print_report", print_report_spy)

    async def target(inputs: str) -> str:
        return inputs

    with pytest.warns(RuntimeWarning, match="received no knowledge base"):
        result = await quality_scan(
            target=target,
            description="Support agent",
            languages=["en"],
        )

    assert result.results == []
    assert result.recommendation == ""
    assert printed_reports == ["component"]


async def test_quality_scan_warns_and_skips_empty_raw_knowledge_base(
    monkeypatch: pytest.MonkeyPatch,
):
    quality_suite_generator_registry.register(HallucinationScenarioGenerator())
    quality_suite_generator_registry.register(SycophancyScenarioGenerator())
    quality_suite_generator_registry.register(SplitQuestionsScenarioGenerator())
    quality_suite_generator_registry.register(MultiTopicScenarioGenerator())
    quality_suite_generator_registry.register(OutOfScopeScenarioGenerator())
    captured_generators: list[ScenarioGenerator] = []
    captured_knowledge_bases: list[object] = []

    class _FakeSuite:
        async def run(
            self,
            target: object,
            parallel: bool = True,
            max_concurrency: int | None = None,
            return_exception: bool = False,
        ) -> SuiteResult:
            _ = target, parallel, max_concurrency, return_exception
            return SuiteResult(results=[], duration_ms=0)

    async def generate_suite_spy(**kwargs: Any) -> _FakeSuite:
        captured_generators.extend(kwargs["generators"])
        captured_knowledge_bases.append(kwargs["knowledge_base"])
        return _FakeSuite()

    monkeypatch.setattr(quality_module, "generate_suite", generate_suite_spy)

    async def target(inputs: str) -> str:
        return inputs

    with pytest.warns(RuntimeWarning, match="received an empty knowledge base"):
        await quality_scan(
            target=target,
            description="Support agent",
            languages=["en"],
            knowledge_base=["", "  "],
        )

    assert len(captured_generators) == 5
    assert captured_knowledge_bases == [None]
    assert {type(generator) for generator in captured_generators} == {
        HallucinationScenarioGenerator,
        SycophancyScenarioGenerator,
        SplitQuestionsScenarioGenerator,
        MultiTopicScenarioGenerator,
        OutOfScopeScenarioGenerator,
    }


async def test_quality_scan_configures_knowledge_base_generator(
    monkeypatch: pytest.MonkeyPatch,
):
    quality_suite_generator_registry.register(HallucinationScenarioGenerator())
    quality_suite_generator_registry.register(SycophancyScenarioGenerator())
    quality_suite_generator_registry.register(SplitQuestionsScenarioGenerator())
    quality_suite_generator_registry.register(MultiTopicScenarioGenerator())
    quality_suite_generator_registry.register(OutOfScopeScenarioGenerator())
    captured_generators: list[ScenarioGenerator] = []
    captured_knowledge_bases: list[object] = []
    printed_reports: list[str | None] = []

    class _FakeSuite:
        async def run(
            self,
            target: object,
            parallel: bool = True,
            max_concurrency: int | None = None,
            return_exception: bool = False,
        ) -> SuiteResult:
            _ = target, parallel, max_concurrency, return_exception
            return SuiteResult(results=[], duration_ms=0)

    async def generate_suite_spy(**kwargs: Any) -> _FakeSuite:
        captured_generators.extend(kwargs["generators"])
        captured_knowledge_bases.append(kwargs["knowledge_base"])
        return _FakeSuite()

    monkeypatch.setattr(quality_module, "generate_suite", generate_suite_spy)
    monkeypatch.setattr(
        SuiteResult,
        "print_report",
        lambda self, group_by=None: printed_reports.append(group_by),
    )

    async def target(inputs: str) -> str:
        return inputs

    await quality_scan(
        target=target,
        description="Support agent",
        languages=["en"],
        knowledge_base=["alpha"],
    )

    assert printed_reports == ["component"]
    assert len(captured_generators) == 5
    assert len(captured_knowledge_bases) == 1
    assert isinstance(captured_knowledge_bases[0], KnowledgeBase)
    assert [document.content for document in captured_knowledge_bases[0].documents] == [
        "alpha"
    ]
    assert {type(generator) for generator in captured_generators} == {
        HallucinationScenarioGenerator,
        SycophancyScenarioGenerator,
        SplitQuestionsScenarioGenerator,
        MultiTopicScenarioGenerator,
        OutOfScopeScenarioGenerator,
    }


async def test_quality_scan_does_not_generate_recommendation_without_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    quality_suite_generator_registry.register(_DeterministicQualityGenerator())
    recommendation_generator = _MockRecommendationGenerator(
        responses=["This should not be used."]
    )
    monkeypatch.setattr(settings, "_default_generator", recommendation_generator)
    monkeypatch.setattr(SuiteResult, "print_report", lambda self, **_: None)

    async def target(inputs: str) -> str:
        return inputs

    result = await quality_scan(
        target=target,
        description="Support agent",
        languages=["en"],
        knowledge_base=["reference document"],
        max_scenarios=1,
    )

    assert result.recommendation == ""
    assert recommendation_generator.calls == []


async def test_quality_scan_hides_recommendation_when_generation_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    quality_suite_generator_registry.register(_DeterministicQualityGenerator())
    monkeypatch.setattr(SuiteResult, "print_report", lambda self, **_: None)

    async def fail_recommendation(_: SuiteResult) -> str:
        raise RuntimeError("generator unavailable")

    monkeypatch.setattr(
        quality_module, "generate_quality_recommendation", fail_recommendation
    )

    async def target(inputs: str) -> str:
        return inputs

    result = await quality_scan(
        target=target,
        description="Support agent",
        languages=["en"],
        knowledge_base=["reference document"],
        max_scenarios=2,
    )

    assert result.failed_count == 1
    assert result.recommendation == ""
