import json

import pytest
from giskard.scan.generators import huggingface as hf_mod
from giskard.scan.generators.huggingface import (
    HuggingFaceDatasetScenarioGenerator,
    _resolve_data_files,
)


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
        {"split": "test"},  # missing path
        "not-a-dict",
        {"split": "test", "path": 123},  # non-string path
    ]
    assert _resolve_data_files(data_files) == ["ok.jsonl"]


def test_resolve_data_files_none_or_empty():
    assert _resolve_data_files(None) == []
    assert _resolve_data_files([]) == []


def _scenario_line(name: str) -> str:
    return json.dumps({"name": name, "steps": [], "annotations": {}})


@pytest.fixture
def hf_repo(tmp_path, monkeypatch):
    """Fake a HF dataset repo: filename -> jsonl content on disk.

    Patches list_repo_files and hf_hub_download in the generator module so no
    network access happens.
    """
    files: dict[str, str] = {}

    def add_file(repo_file: str, *names: str) -> None:
        path = tmp_path / repo_file
        path.write_text("\n".join(_scenario_line(n) for n in names) + "\n")
        files[repo_file] = str(path)

    monkeypatch.setattr(
        hf_mod, "list_repo_files", lambda repo_id, repo_type=None: list(files)
    )
    monkeypatch.setattr(
        hf_mod,
        "hf_hub_download",
        lambda repo_id, repo_file, repo_type=None: files[repo_file],
    )
    # _list_available_languages is process-cached by (repo_id, filename); tests
    # reuse the same repo_id/filename with different file sets, so clear it.
    hf_mod._list_available_languages.cache_clear()
    return add_file


def _make_gen(**kwargs) -> HuggingFaceDatasetScenarioGenerator:
    return HuggingFaceDatasetScenarioGenerator(
        repo_id="org/dataset",
        filename="donotanswer.{language}.jsonl",
        **kwargs,
    )


def test_allow_commercial_use_reflects_repo_field():
    assert _make_gen(repo_allow_commercial_use=False).allow_commercial_use is False
    assert _make_gen(repo_allow_commercial_use=True).allow_commercial_use is True


def test_allow_commercial_use_defaults_true():
    assert _make_gen().allow_commercial_use is True


async def test_loads_single_requested_language(hf_repo):
    hf_repo("donotanswer.en.jsonl", "en1", "en2")
    gen = _make_gen()
    scenarios = await gen.generate_scenario("desc", ["en"])
    assert [s.name for s in scenarios] == ["en1", "en2"]


async def test_returns_all_compatible_languages(hf_repo):
    hf_repo("donotanswer.en.jsonl", "en1")
    hf_repo("donotanswer.fr.jsonl", "fr1")
    gen = _make_gen()
    scenarios = await gen.generate_scenario("desc", ["fr", "en"])
    assert sorted(s.name for s in scenarios) == ["en1", "fr1"]


async def test_skips_incompatible_language(hf_repo):
    hf_repo("donotanswer.en.jsonl", "en1")
    gen = _make_gen()
    scenarios = await gen.generate_scenario("desc", ["en", "xx"])
    assert [s.name for s in scenarios] == ["en1"]


async def test_no_compatible_language_returns_empty_and_warns(hf_repo, caplog):
    hf_repo("donotanswer.en.jsonl", "en1")
    gen = _make_gen()
    with caplog.at_level("WARNING"):
        scenarios = await gen.generate_scenario("desc", ["xx", "yy"])
    assert scenarios == []
    assert "No compatible language" in caplog.text


async def test_injects_description_and_languages(hf_repo):
    hf_repo("donotanswer.en.jsonl", "en1")
    gen = _make_gen()
    scenarios = await gen.generate_scenario("my agent", ["en", "fr"])
    assert scenarios[0].annotations["description"] == "my agent"
    assert scenarios[0].annotations["languages"] == ["en", "fr"]


async def test_applies_tags(hf_repo):
    hf_repo("donotanswer.en.jsonl", "en1")
    gen = _make_gen(tags=["dataset:do-not-answer"])
    scenarios = await gen.generate_scenario("desc", ["en"])
    assert scenarios[0].tags == ["dataset:do-not-answer"]


async def test_malformed_jsonl_raises_with_source(hf_repo, tmp_path, monkeypatch):
    bad = tmp_path / "donotanswer.en.jsonl"
    bad.write_text('{"name": "ok", "steps": [], "annotations": {}}\n{bad\n')
    monkeypatch.setattr(
        hf_mod,
        "list_repo_files",
        lambda repo_id, repo_type=None: ["donotanswer.en.jsonl"],
    )
    monkeypatch.setattr(
        hf_mod, "hf_hub_download", lambda repo_id, repo_file, repo_type=None: str(bad)
    )
    hf_mod._list_available_languages.cache_clear()
    gen = _make_gen()
    with pytest.raises(ValueError, match=r"org/dataset/donotanswer\.en\.jsonl|line 2"):
        await gen.generate_scenario("desc", ["en"])
