import os

import pytest
from giskard import agents


@pytest.fixture
async def generator():
    """Fixture providing a configured generator for tests."""
    return agents.Generator(model=os.getenv("TEST_MODEL", "gemini/gemini-2.0-flash"))
