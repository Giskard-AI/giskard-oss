from giskard.checks import Suite
from giskard.checks.scenarios.catalog import ScenarioCategory, generate_suite


def test_scenario_category_enum_has_llm01():
    assert ScenarioCategory.LLM01_INDIRECT_INJECTION == "llm01_indirect_injection"


def test_generate_suite_returns_suite():
    suite = generate_suite(
        categories=[ScenarioCategory.LLM01_INDIRECT_INJECTION],
        description="A documentation chatbot for Giskard",
    )
    assert isinstance(suite, Suite)


def test_generate_suite_loads_scenarios():
    suite = generate_suite(
        categories=[ScenarioCategory.LLM01_INDIRECT_INJECTION],
        description="A test agent",
    )
    assert len(suite.scenarios) >= 1


def test_generate_suite_max_scenarios_limits_count():
    # Only applies if there are more scenarios than max_scenarios
    # With 1 JSONL entry, max_scenarios=1 should return 1
    suite = generate_suite(
        categories=[ScenarioCategory.LLM01_INDIRECT_INJECTION],
        description="A test agent",
        max_scenarios=1,
    )
    assert len(suite.scenarios) == 1


def test_generate_suite_max_scenarios_none_returns_all():
    suite = generate_suite(
        categories=[ScenarioCategory.LLM01_INDIRECT_INJECTION],
        description="A test agent",
        max_scenarios=None,
    )
    assert len(suite.scenarios) >= 1


def test_generate_suite_is_reproducible_with_same_seed(monkeypatch):
    # Patch _load_scenarios to return a pool larger than max_scenarios so that
    # seeding actually affects selection (with 1 JSONL entry, any seed returns the same result).
    from giskard.checks.core.scenario import Scenario
    from giskard.checks.scenarios import catalog

    fake_pool = [Scenario(name=f"scenario_{i}") for i in range(5)]
    monkeypatch.setattr(catalog, "_load_scenarios", lambda _cat: list(fake_pool))

    suite_a = generate_suite(
        categories=[ScenarioCategory.LLM01_INDIRECT_INJECTION],
        description="A test agent",
        max_scenarios=3,
        seed=42,
    )
    suite_b = generate_suite(
        categories=[ScenarioCategory.LLM01_INDIRECT_INJECTION],
        description="A test agent",
        max_scenarios=3,
        seed=42,
    )
    assert [s.name for s in suite_a.scenarios] == [s.name for s in suite_b.scenarios]


def test_generate_suite_different_seeds_give_different_results(monkeypatch):
    from giskard.checks.core.scenario import Scenario
    from giskard.checks.scenarios import catalog

    fake_pool = [Scenario(name=f"scenario_{i}") for i in range(10)]
    monkeypatch.setattr(catalog, "_load_scenarios", lambda _cat: list(fake_pool))

    suite_a = generate_suite(
        categories=[ScenarioCategory.LLM01_INDIRECT_INJECTION],
        description="A test agent",
        max_scenarios=5,
        seed=1,
    )
    suite_b = generate_suite(
        categories=[ScenarioCategory.LLM01_INDIRECT_INJECTION],
        description="A test agent",
        max_scenarios=5,
        seed=99,
    )
    assert [s.name for s in suite_a.scenarios] != [s.name for s in suite_b.scenarios]


def test_generate_suite_injects_description_as_annotation():
    description = "A customer support chatbot for an e-commerce platform"
    suite = generate_suite(
        categories=[ScenarioCategory.LLM01_INDIRECT_INJECTION],
        description=description,
    )
    for scenario in suite.scenarios:
        assert scenario.annotations.get("description") == description


def test_generate_suite_suite_has_no_target():
    from giskard.core.utils import NotProvided

    suite = generate_suite(
        categories=[ScenarioCategory.LLM01_INDIRECT_INJECTION],
        description="A test agent",
    )
    assert isinstance(suite.target, NotProvided)


def test_generate_suite_custom_name():
    suite = generate_suite(
        categories=[ScenarioCategory.LLM01_INDIRECT_INJECTION],
        description="A test agent",
        name="My Custom Suite",
    )
    assert suite.name == "My Custom Suite"


def test_generate_suite_default_name():
    suite = generate_suite(
        categories=[ScenarioCategory.LLM01_INDIRECT_INJECTION],
        description="A test agent",
    )
    assert suite.name == "Security Suite"


def test_generate_suite_preserves_multiple_runs():
    suite = generate_suite(
        categories=[ScenarioCategory.LLM01_INDIRECT_INJECTION],
        description="A test agent",
    )
    # The LLM01 JSONL entry has multiple_runs=5
    for scenario in suite.scenarios:
        assert scenario.multiple_runs == 5
