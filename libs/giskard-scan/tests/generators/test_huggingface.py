import json
from types import SimpleNamespace
from typing import Any

import pytest
from giskard.scan.generators import huggingface as hf_mod
from giskard.scan.generators.base import ScenarioContext
from giskard.scan.generators.huggingface import (
    HuggingFaceDatasetScenarioGenerator,
    _resolve_data_files,
)


def _scenario_line(name: str) -> str:
    return json.dumps({"name": name, "steps": [], "annotations": {}})


# --- _resolve_data_files (pure) -------------------------------------------------


def test_resolve_data_files_single_entry():
    assert _resolve_data_files([{"split": "test", "path": "donotanswer.en.jsonl"}]) == [
        "donotanswer.en.jsonl"
    ]


def test_resolve_data_files_multiple_entries():
    data_files = [
        {"split": "test", "path": "a.jsonl"},
        {"split": "test", "path": "b.jsonl"},
    ]
    assert _resolve_data_files(data_files) == ["a.jsonl", "b.jsonl"]


def test_resolve_data_files_skips_malformed_entries():
    data_files = [
        {"split": "test", "path": "ok.jsonl"},
        {"split": "test"},
        "not-a-dict",
        {"split": "test", "path": 123},
    ]
    assert _resolve_data_files(data_files) == ["ok.jsonl"]


def test_resolve_data_files_none_or_empty():
    assert _resolve_data_files(None) == []
    assert _resolve_data_files([]) == []


# --- fake HF repo fixture -------------------------------------------------------


@pytest.fixture
def hf_repo(tmp_path, monkeypatch):
    """Fake a HF dataset repo: write jsonl files + a matching card.

    Patches DatasetCard.load, list_repo_files, and hf_hub_download in the
    generator module so no network access happens. The card's ``configs`` is
    built from registered (config_name -> [repo files]) entries.
    """
    files: dict[str, str] = {}
    configs: list[dict[str, Any]] = []

    def add_subset(config_name: str, repo_file: str, *names: str) -> None:
        """Register one file under a config; appends to the config if it exists."""
        path = tmp_path / repo_file
        path.write_text("\n".join(_scenario_line(n) for n in names) + "\n")
        files[repo_file] = str(path)
        for cfg in configs:
            if cfg["config_name"] == config_name:
                cfg["data_files"].append({"split": "test", "path": repo_file})
                return
        configs.append(
            {
                "config_name": config_name,
                "data_files": [{"split": "test", "path": repo_file}],
            }
        )

    def declare_config(config_name: str, repo_file: str) -> None:
        """Declare a config pointing at a file that is NOT written to the repo."""
        configs.append(
            {
                "config_name": config_name,
                "data_files": [{"split": "test", "path": repo_file}],
            }
        )

    def fake_card_load(repo_id, repo_type=None):
        return SimpleNamespace(data=SimpleNamespace(configs=list(configs)))

    monkeypatch.setattr(hf_mod.DatasetCard, "load", staticmethod(fake_card_load))
    monkeypatch.setattr(
        hf_mod, "list_repo_files", lambda repo_id, repo_type=None: list(files)
    )
    monkeypatch.setattr(
        hf_mod,
        "hf_hub_download",
        lambda repo_id, repo_file, repo_type=None: files[repo_file],
    )
    hf_mod._language_subsets.cache_clear()

    return SimpleNamespace(add_subset=add_subset, declare_config=declare_config)


def _make_gen(**kwargs) -> HuggingFaceDatasetScenarioGenerator:
    return HuggingFaceDatasetScenarioGenerator(repo_id="org/dataset", **kwargs)


def _context(
    description: str = "desc", languages: list[str] | None = None
) -> ScenarioContext:
    return ScenarioContext(description=description, languages=languages or ["en"])


# --- allow_commercial_use -------------------------------------------------------


def test_allow_commercial_use_reflects_repo_field():
    assert _make_gen(repo_allow_commercial_use=False).allow_commercial_use is False
    assert _make_gen(repo_allow_commercial_use=True).allow_commercial_use is True


def test_allow_commercial_use_defaults_true():
    assert _make_gen().allow_commercial_use is True


# --- _language_subsets discovery ------------------------------------------------


def test_language_subsets_maps_present_files(hf_repo):
    hf_repo.add_subset("en", "donotanswer.en.jsonl", "en1")
    assert hf_mod._language_subsets("org/dataset") == {"en": ["donotanswer.en.jsonl"]}


def test_language_subsets_omits_config_with_no_present_files(hf_repo):
    hf_repo.declare_config("fr", "missing.fr.jsonl")
    assert hf_mod._language_subsets("org/dataset") == {}


def test_language_subsets_no_configs_returns_empty(hf_repo):
    assert hf_mod._language_subsets("org/dataset") == {}


# --- load behavior --------------------------------------------------------------


async def test_loads_single_requested_language(hf_repo):
    hf_repo.add_subset("en", "donotanswer.en.jsonl", "en1", "en2")
    gen = _make_gen()
    scenarios = await gen.generate_scenario(_context(languages=["en"]))
    assert [s.name for s in scenarios] == ["en1", "en2"]


async def test_multi_file_language_concatenates(hf_repo):
    hf_repo.add_subset("en", "en.part1.jsonl", "en1")
    hf_repo.add_subset("en", "en.part2.jsonl", "en2")
    gen = _make_gen()
    scenarios = await gen.generate_scenario(_context(languages=["en"]))
    assert sorted(s.name for s in scenarios) == ["en1", "en2"]


async def test_returns_all_compatible_languages(hf_repo):
    hf_repo.add_subset("en", "en.jsonl", "en1")
    hf_repo.add_subset("fr", "fr.jsonl", "fr1")
    gen = _make_gen()
    scenarios = await gen.generate_scenario(_context(languages=["fr", "en"]))
    assert sorted(s.name for s in scenarios) == ["en1", "fr1"]


async def test_skips_incompatible_language(hf_repo):
    hf_repo.add_subset("en", "en.jsonl", "en1")
    gen = _make_gen()
    scenarios = await gen.generate_scenario(_context(languages=["en", "xx"]))
    assert [s.name for s in scenarios] == ["en1"]


async def test_no_compatible_language_returns_empty_and_warns(hf_repo, caplog):
    hf_repo.add_subset("en", "en.jsonl", "en1")
    gen = _make_gen()
    with caplog.at_level("WARNING"):
        scenarios = await gen.generate_scenario(_context(languages=["xx", "yy"]))
    assert scenarios == []
    assert "No compatible language" in caplog.text


async def test_injects_description_and_languages(hf_repo):
    hf_repo.add_subset("en", "en.jsonl", "en1")
    gen = _make_gen()
    scenarios = await gen.generate_scenario(
        _context(description="my agent", languages=["en", "fr"])
    )
    assert scenarios[0].annotations["description"] == "my agent"
    assert scenarios[0].annotations["languages"] == ["en", "fr"]


async def test_applies_tags(hf_repo):
    hf_repo.add_subset("en", "en.jsonl", "en1")
    gen = _make_gen(tags=["dataset:do-not-answer"])
    scenarios = await gen.generate_scenario(_context(languages=["en"]))
    assert scenarios[0].tags == ["dataset:do-not-answer"]


async def test_malformed_jsonl_raises_with_source(hf_repo, tmp_path, monkeypatch):
    bad = tmp_path / "donotanswer.en.jsonl"
    bad.write_text('{"name": "ok", "steps": [], "annotations": {}}\n{bad\n')
    monkeypatch.setattr(
        hf_mod.DatasetCard,
        "load",
        staticmethod(
            lambda repo_id, repo_type=None: SimpleNamespace(
                data=SimpleNamespace(
                    configs=[
                        {
                            "config_name": "en",
                            "data_files": [
                                {"split": "test", "path": "donotanswer.en.jsonl"}
                            ],
                        }
                    ]
                )
            )
        ),
    )
    monkeypatch.setattr(
        hf_mod,
        "list_repo_files",
        lambda repo_id, repo_type=None: ["donotanswer.en.jsonl"],
    )
    monkeypatch.setattr(
        hf_mod, "hf_hub_download", lambda repo_id, repo_file, repo_type=None: str(bad)
    )
    hf_mod._language_subsets.cache_clear()
    gen = _make_gen()
    with pytest.raises(ValueError, match=r"org/dataset/donotanswer\.en\.jsonl|line 2"):
        await gen.generate_scenario(_context(languages=["en"]))
