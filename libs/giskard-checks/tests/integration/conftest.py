import pytest

# Mark every test in this directory as integration by default.
# Opt out of GISKARD_* isolation so ambient GISKARD_CHECKS_DEFAULT_MODEL
# (and related) still select the generator for --run-integration runs.
pytestmark = [pytest.mark.integration, pytest.mark.uses_ambient_env]
