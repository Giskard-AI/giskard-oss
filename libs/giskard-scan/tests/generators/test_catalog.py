import asyncio
from typing import Any

import numpy as np
import pytest
from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from giskard.scan.catalog import generate_suite
from giskard.scan.generators.base import ScenarioContext, ScenarioGenerator
from giskard.scan.utils.knowledge_base import KnowledgeBase
from giskard.scan.vulnerability import vulnerability_suite_generator_registry


class _ModeTracker(ScenarioGenerator):
    """Records the last target_mode seen by each named instance."""

    name: str
    seen_mode: str = ""

    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng=None,
        target_mode: str = "multiturn",
    ):
        self.seen_mode = target_mode
        return []


class _StubGenerator(ScenarioGenerator):
    """Returns a fixed number of scenarios for testing."""

    name: str = "stub"
    scenario_count: int = 1

    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
        target_mode: str = "multiturn",
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        n = max_scenarios if max_scenarios is not None else self.scenario_count
        return [Scenario(name=f"stub-{self.name}-{i}") for i in range(n)]


async def test_generate_suite_requires_explicit_generators():
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'generators'"
    ):
        await generate_suite(  # pyright: ignore[reportCallIssue]
            "My chatbot", languages=["en"]
        )


async def test_generate_suite_does_not_read_vulnerability_registry(
    isolated_vulnerability_registry,
):
    vulnerability_suite_generator_registry.register(_StubGenerator(name="registry"))
    suite = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_StubGenerator(name="explicit")],
    )
    assert [scenario.name for scenario in suite.scenarios] == ["stub-explicit-0"]


async def test_generate_suite_generators_bare_type_is_normalized():
    suite = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_StubGenerator],
    )
    assert len(suite.scenarios) == 1
    assert suite.scenarios[0].name == "stub-stub-0"


async def test_generate_suite_empty_generators_returns_empty_suite(
    isolated_vulnerability_registry,
):
    vulnerability_suite_generator_registry.register(_StubGenerator(name="a"))
    suite = await generate_suite("My chatbot", languages=["en"], generators=[])
    assert suite.scenarios == []


async def test_generate_suite_max_scenarios_limits_output():
    suite = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[
            _StubGenerator(name="a", scenario_count=5),
            _StubGenerator(name="b", scenario_count=5),
        ],
        max_scenarios=2,
    )
    assert len(suite.scenarios) == 2


async def test_generate_suite_max_scenarios_distributed_across_generators():
    """Budget is split across generators; each receives a non-negative count summing to max_scenarios."""
    received: dict[str, int | None] = {}

    class _TrackingGenerator(ScenarioGenerator):
        name: str

        async def generate_scenario(
            self,
            context: ScenarioContext,
            max_scenarios: int | None = None,
            rng: np.random.Generator | None = None,
            target_mode: str = "multiturn",
        ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
            received[self.name] = max_scenarios
            n = max_scenarios if max_scenarios is not None else 0
            return [Scenario(name=f"{self.name}-{i}") for i in range(n)]

    await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_TrackingGenerator(name="x"), _TrackingGenerator(name="y")],
        max_scenarios=4,
        seed=42,
    )
    assert received["x"] is not None
    assert received["y"] is not None
    assert received["x"] + received["y"] == 4


async def test_generate_suite_no_max_passes_none_to_generators():
    """Without max_scenarios, generators receive max_scenarios=None."""
    received: dict[str, int | None] = {}

    class _TrackingGenerator(ScenarioGenerator):
        name: str

        async def generate_scenario(
            self,
            context: ScenarioContext,
            max_scenarios: int | None = None,
            rng: np.random.Generator | None = None,
            target_mode: str = "multiturn",
        ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
            received[self.name] = max_scenarios
            return []

    await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_TrackingGenerator(name="z")],
    )
    assert received["z"] is None


async def test_generate_suite_forwards_knowledge_base_to_generators():
    received: list[object] = []

    class _KnowledgeBaseTrackingGenerator(ScenarioGenerator):
        async def generate_scenario(
            self,
            context: ScenarioContext,
            max_scenarios: int | None = None,
            rng: np.random.Generator | None = None,
            target_mode: str = "multiturn",
        ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
            received.append(context.knowledge_base)
            return []

    await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_KnowledgeBaseTrackingGenerator()],
        knowledge_base=["reference document"],
    )

    assert len(received) == 1
    kb = received[0]
    assert isinstance(kb, KnowledgeBase)
    assert [doc.content for doc in kb.documents] == ["reference document"]


async def test_generate_suite_passes_context_to_all_generators():
    seen: list[str] = []

    class _ContextReader(ScenarioGenerator):
        name: str = "reader"

        async def generate_scenario(
            self,
            context: ScenarioContext,
            max_scenarios: int | None = None,
            rng: np.random.Generator | None = None,
            target_mode: str = "multiturn",
        ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
            seen.append(context.description)
            return []

    await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_ContextReader(), _ContextReader()],
    )

    assert seen == ["My chatbot", "My chatbot"]


async def test_generate_suite_registry_generators_not_mutated():
    """The catalog must not mutate provided generator instances."""
    gen = _StubGenerator(name="orig", scenario_count=3)
    original_count = gen.scenario_count

    await generate_suite(
        "My chatbot", languages=["en"], generators=[gen], max_scenarios=1
    )

    assert gen.scenario_count == original_count


async def test_generate_suite_negative_max_scenarios_raises_valueerror():
    """max_scenarios < 0 raises ValueError."""
    with pytest.raises(ValueError, match="max_scenarios must be non-negative, got -1"):
        await generate_suite(
            "My chatbot", languages=["en"], generators=[], max_scenarios=-1
        )


async def test_generate_suite_max_scenarios_zero_returns_empty():
    """max_scenarios=0 is a valid no-op budget: returns an empty suite."""
    suite = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_StubGenerator(name="a", scenario_count=5)],
        max_scenarios=0,
    )
    assert suite.scenarios == []


async def test_generate_suite_non_kb_generator_ignores_knowledge_base():
    """A generator that does not use context.knowledge_base still runs normally."""

    class _KbIgnoringGenerator(ScenarioGenerator):
        async def generate_scenario(
            self,
            context: ScenarioContext,
            max_scenarios: int | None = None,
            rng: np.random.Generator | None = None,
            target_mode: str = "multiturn",
        ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
            # Deliberately never reads context.knowledge_base.
            return [Scenario(name="kb-ignored")]

    suite = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_KbIgnoringGenerator()],
        knowledge_base=["some doc"],
    )

    assert len(suite.scenarios) == 1
    assert suite.scenarios[0].name == "kb-ignored"


async def test_generate_suite_reproducibility():
    """Same seed produces identical per-generator scenario name allocation."""

    class _TrackingGenerator(ScenarioGenerator):
        name: str

        async def generate_scenario(
            self,
            context: ScenarioContext,
            max_scenarios: int | None = None,
            rng: np.random.Generator | None = None,
            target_mode: str = "multiturn",
        ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
            return [
                Scenario(name=f"{self.name}-{i}") for i in range(max_scenarios or 0)
            ]

    generators = [_TrackingGenerator(name="p"), _TrackingGenerator(name="q")]

    suite_a = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=generators,
        max_scenarios=10,
        seed=99,
    )
    suite_b = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=generators,
        max_scenarios=10,
        seed=99,
    )

    assert [s.name for s in suite_a.scenarios] == [s.name for s in suite_b.scenarios]


async def test_generate_suite_passes_target_mode_to_generators():
    """target_mode is forwarded to each generator's generate_scenario."""
    tracker_a = _ModeTracker(name="a")
    tracker_b = _ModeTracker(name="b")

    await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[tracker_a, tracker_b],
        target_mode="singleturn",
    )
    assert tracker_a.seen_mode == "singleturn"
    assert tracker_b.seen_mode == "singleturn"


async def test_generate_suite_target_mode_defaults_to_multiturn():
    tracker = _ModeTracker(name="z")

    await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[tracker],
    )
    assert tracker.seen_mode == "multiturn"


async def test_generate_suite_singleturn_passes_scenarios_through():
    """In singleturn mode, scenarios from generators are passed through unchanged."""
    single = Scenario(name="ok").interact("hello", outputs=lambda inputs: "world")

    class _SingleGen(ScenarioGenerator):
        async def generate_scenario(
            self,
            context: ScenarioContext,
            max_scenarios=None,
            rng=None,
            target_mode="multiturn",
        ):
            return [single]

    suite = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_SingleGen()],
        target_mode="singleturn",
    )
    assert len(suite.scenarios) == 1
    assert suite.scenarios[0].name == "ok"


async def test_generate_suite_unbudgeted_reproducibility():
    """Without max_scenarios, each generator gets an independent child rng so
    concurrent draws stay reproducible across runs with the same seed."""

    class _RngGenerator(ScenarioGenerator):
        name: str
        # Yield to the event loop before drawing so that, with a *shared* rng,
        # draw order would depend on coroutine scheduling. Independent child
        # rngs make each generator's draw stable regardless of interleaving.
        yield_seconds: float

        async def generate_scenario(
            self,
            context: ScenarioContext,
            max_scenarios: int | None = None,
            rng: np.random.Generator | None = None,
            target_mode: str = "multiturn",
        ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
            assert rng is not None
            await asyncio.sleep(self.yield_seconds)
            draw = int(rng.integers(0, 1_000_000))
            return [Scenario(name=f"{self.name}-{draw}")]

    # Different delays force the two coroutines to draw in a fixed wall-clock
    # order, which is the order a shared stream would couple them to.
    generators = [
        _RngGenerator(name="p", yield_seconds=0.02),
        _RngGenerator(name="q", yield_seconds=0.0),
    ]

    suite_a = await generate_suite(
        "My chatbot", languages=["en"], generators=generators, seed=7
    )

    # Same generators, draws now reversed by swapping the delays. With a shared
    # rng this flips which generator gets which number; independent child rngs
    # keep each generator's number pinned to its identity.
    generators[0].yield_seconds, generators[1].yield_seconds = (0.0, 0.02)
    suite_b = await generate_suite(
        "My chatbot", languages=["en"], generators=generators, seed=7
    )

    names_a = {s.name for s in suite_a.scenarios}
    names_b = {s.name for s in suite_b.scenarios}
    assert names_a == names_b
    # The two generators must draw from independent streams, not the same value.
    draws = sorted(name.split("-")[1] for name in names_a)
    assert draws[0] != draws[1]
