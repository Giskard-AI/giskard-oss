from collections import Counter

import numpy as np
import pytest
from giskard.checks.generators import LLMGenerator
from giskard.checks.judges import Conformity
from giskard.checks.scenarios_generator.adversarial_generator import (
    ADVERSARIAL_CATEGORIES,
    AdversarialCategory,
    AdversarialScenarioGenerator,
)

from ..testing_utils import MockGenerator


def _rules_response(*rules: str) -> dict[str, list[str]]:
    return {"rules": list(rules)}


def _make_generator(*rules_per_call: tuple[str, ...]) -> MockGenerator:
    """Build a MockGenerator returning one rules batch per call."""
    return MockGenerator(
        responses=[_rules_response(*rules) for rules in rules_per_call]
    )


# ---------------------------------------------------------------------------
# _generate_rules
# ---------------------------------------------------------------------------


async def test_generate_rules_returns_exact_count():
    gen = AdversarialScenarioGenerator(
        generator=_make_generator(("rule A", "rule B", "rule C"))
    )
    category = AdversarialCategory(name="Test", description=None)
    rules = await gen._generate_rules(category, description="A chatbot", num_rules=3)
    assert rules == ["rule A", "rule B", "rule C"]


async def test_generate_rules_retries_when_under_generated():
    # First call returns 2 rules, second call returns 1 more — should merge to 3.
    gen = AdversarialScenarioGenerator(
        generator=_make_generator(("rule A", "rule B"), ("rule C",))
    )
    category = AdversarialCategory(name="Test", description=None)
    rules = await gen._generate_rules(category, description="A chatbot", num_rules=3)
    assert rules == ["rule A", "rule B", "rule C"]
    assert (
        isinstance(gen.generator, MockGenerator) and gen.generator.index == 2
    )  # two LLM calls were made


async def test_generate_rules_caps_at_num_rules_when_over_generated():
    gen = AdversarialScenarioGenerator(
        generator=_make_generator(("rule A", "rule B", "rule C", "rule D"))
    )
    category = AdversarialCategory(name="Test", description=None)
    rules = await gen._generate_rules(category, description="A chatbot", num_rules=2)
    assert rules == ["rule A", "rule B"]


async def test_generate_rules_stops_after_max_retries():
    # Simulate a generator that always returns fewer rules than asked.
    gen = AdversarialScenarioGenerator(
        generator=_make_generator(("only one",), ("only one",), ("only one",))
    )
    category = AdversarialCategory(name="Test", description=None)
    # After 3 attempts we get 3 rules, but num_rules=5 → capped at 3
    rules = await gen._generate_rules(category, description="A chatbot", num_rules=5)
    assert len(rules) == 3
    assert isinstance(gen.generator, MockGenerator) and gen.generator.index == 3


# ---------------------------------------------------------------------------
# generate_scenario — shape and annotations
# ---------------------------------------------------------------------------


@pytest.fixture
def scenario_generator() -> AdversarialScenarioGenerator:
    """Generator that returns 5 rules per category in a single LLM call each."""
    num_categories = len(ADVERSARIAL_CATEGORIES)
    rules = [f"rule {i}" for i in range(1, 6)]
    responses = [_rules_response(*rules)] * num_categories
    return AdversarialScenarioGenerator(generator=MockGenerator(responses=responses))


async def test_generate_scenario_returns_correct_count(
    scenario_generator: AdversarialScenarioGenerator,
):
    scenarios = await scenario_generator.generate_scenario(
        description="A chatbot", languages=["en"]
    )
    # 8 categories × 5 rules each = 40 scenarios
    assert len(scenarios) == len(ADVERSARIAL_CATEGORIES) * 5


async def test_generate_scenario_names_include_category(
    scenario_generator: AdversarialScenarioGenerator,
):
    scenarios = await scenario_generator.generate_scenario(
        description="A chatbot", languages=["en"]
    )
    category_names = {c.name for c in ADVERSARIAL_CATEGORIES}
    for scenario in scenarios:
        assert any(name in scenario.name for name in category_names)


async def test_generate_scenario_annotations_contain_expected_keys(
    scenario_generator: AdversarialScenarioGenerator,
):
    scenarios = await scenario_generator.generate_scenario(
        description="My chatbot", languages=["en", "fr"]
    )
    for scenario in scenarios:
        ann = scenario.annotations
        assert ann["description"] == "My chatbot"
        assert set(ann["languages"]) == {"en", "fr"}
        assert "category" in ann
        assert "name" in ann["category"]
        assert "rule" in ann


async def test_generate_scenario_each_scenario_has_interact_and_check(
    scenario_generator: AdversarialScenarioGenerator,
):
    scenarios = await scenario_generator.generate_scenario(
        description="A chatbot", languages=["en"]
    )
    for scenario in scenarios:
        assert len(scenario.steps) >= 1
        step = scenario.steps[0]
        assert len(step.interacts) == 1
        assert isinstance(step.interacts[0], LLMGenerator)
        assert len(step.checks) == 1
        assert isinstance(step.checks[0], Conformity)


async def test_generate_scenario_rule_in_conformity_check(
    scenario_generator: AdversarialScenarioGenerator,
):
    scenarios = await scenario_generator.generate_scenario(
        description="A chatbot", languages=["en"]
    )
    for scenario in scenarios:
        conformity = scenario.steps[0].checks[0]
        assert isinstance(conformity, Conformity)
        assert conformity.rule in {f"rule {i}" for i in range(1, 6)}


async def test_generate_scenario_makes_one_llm_call_per_category(
    scenario_generator: AdversarialScenarioGenerator,
):
    await scenario_generator.generate_scenario(
        description="A chatbot", languages=["en"]
    )
    assert isinstance(
        scenario_generator.generator, MockGenerator
    ) and scenario_generator.generator.index == len(ADVERSARIAL_CATEGORIES)


# ---------------------------------------------------------------------------
# generate_scenario — budget (max_scenarios)
# ---------------------------------------------------------------------------


async def test_generate_scenario_with_budget_skips_zero_rule_categories():
    """Categories assigned 0 rules by multinomial are skipped (no LLM calls)."""
    rng = np.random.default_rng(0)
    n_cats = len(ADVERSARIAL_CATEGORIES)
    responses = [_rules_response(f"rule {i}") for i in range(n_cats)]
    gen = AdversarialScenarioGenerator(generator=MockGenerator(responses=responses))
    scenarios = await gen.generate_scenario(
        description="A chatbot", languages=["en"], max_scenarios=1, rng=rng
    )
    # With budget=1, at most 1 scenario is generated
    assert len(scenarios) <= 1
    # Number of LLM calls equals number of non-zero-budget categories
    assert isinstance(gen.generator, MockGenerator)
    assert gen.generator.index < n_cats  # fewer calls than full run


async def test_generate_scenario_with_budget_caps_rules_per_category():
    """Rules per category are capped at MAX_RULES_PER_CATEGORY even if budget is large."""
    from giskard.checks.scenarios_generator.adversarial_generator import (
        MAX_RULES_PER_CATEGORY,
    )

    n_cats = len(ADVERSARIAL_CATEGORIES)
    rng = np.random.default_rng(42)
    responses = [
        _rules_response(*[f"rule {i}" for i in range(MAX_RULES_PER_CATEGORY)])
    ] * n_cats
    gen = AdversarialScenarioGenerator(generator=MockGenerator(responses=responses))
    scenarios = await gen.generate_scenario(
        description="A chatbot",
        languages=["en"],
        max_scenarios=n_cats * MAX_RULES_PER_CATEGORY * 2,  # way over cap
        rng=rng,
    )
    cat_counts = Counter(s.annotations["category"]["name"] for s in scenarios)
    for count in cat_counts.values():
        assert count <= MAX_RULES_PER_CATEGORY


async def test_generate_scenario_no_budget_uses_default_rules_per_category():
    """Without a budget, DEFAULT_RULES_PER_CATEGORY rules are requested per category."""
    from giskard.checks.scenarios_generator.adversarial_generator import (
        DEFAULT_RULES_PER_CATEGORY,
    )

    n_cats = len(ADVERSARIAL_CATEGORIES)
    rules = [f"rule {i}" for i in range(1, DEFAULT_RULES_PER_CATEGORY + 1)]
    responses = [_rules_response(*rules)] * n_cats
    gen = AdversarialScenarioGenerator(generator=MockGenerator(responses=responses))
    scenarios = await gen.generate_scenario(description="A chatbot", languages=["en"])
    assert len(scenarios) == n_cats * DEFAULT_RULES_PER_CATEGORY
