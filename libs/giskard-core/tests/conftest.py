import os

import giskard.core.settings as settings_module
import pytest
from giskard.core import disable_telemetry

GISKARD_ENV_PREFIX = "GISKARD_"


@pytest.fixture(autouse=True)
def isolate_giskard_env(monkeypatch: pytest.MonkeyPatch):
    """Isolate tests from ambient ``GISKARD_*`` configuration.

    Removes every ``GISKARD_``-prefixed environment variable and disables the
    ``.env`` file lookup so that tests observe the built-in defaults regardless
    of the developer's local environment (see issue #2734).
    """
    for name in [k for k in os.environ if k.startswith(GISKARD_ENV_PREFIX)]:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setitem(
        settings_module.GiskardCoreSettings.model_config, "env_file", None
    )
    yield


def pytest_configure(config: pytest.Config) -> None:
    """Disable telemetry for tests."""
    disable_telemetry()


def pytest_sessionfinish(session, exitstatus):
    # If no tests were collected, set the exit status to 0 to avoid failure.
    # This is a workaround for packages not having any functional tests.
    if exitstatus == 5:
        session.exitstatus = 0
