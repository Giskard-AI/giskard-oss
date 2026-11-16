import os
import subprocess
import sys

import pytest

HEAVY = [
    "numpy",
    "griffe",
    "posthog",
    "giskard.agents.workflow",
    "giskard.agents.generators",
    "giskard.agents.tools",
    "giskard.agents.embeddings",
]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "DO_NOT_TRACK": "1"}
    return subprocess.run(
        [sys.executable, "-I", "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


def test_templates_import_skips_heavy_modules():
    result = _run(
        f"import sys, giskard.agents.templates\n"
        f"print([m for m in {HEAVY!r} if m in sys.modules])"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_public_names_still_resolve():
    import giskard.agents as ga

    for name in ga.__all__:
        assert getattr(ga, name) is not None
    assert {"ChatWorkflow", "Generator", "MessageTemplate"} <= set(dir(ga))
    assert {"__version__", "__name__"} <= set(dir(ga))


def test_lazy_table_matches_all():
    import giskard.agents as ga

    assert set(ga._LAZY) == set(ga.__all__) - {"__version__"}


def test_submodule_attribute_access_stays_lazy():
    result = _run(
        "import sys, giskard.agents as a\n"
        "assert 'giskard.agents.generators' not in sys.modules\n"
        "a.generators.BaseGenerator; a.workflow.ChatWorkflow\n"
        "print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_unknown_name_raises_attribute_error():
    import giskard.agents as ga

    with pytest.raises(AttributeError):
        ga.NoSuchName
