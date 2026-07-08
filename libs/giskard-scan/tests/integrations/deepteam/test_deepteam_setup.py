"""Availability gate + kwarg-rejection + singleturn early-return tests."""

import pytest
from giskard.scan.integrations.deepteam._adapter import (
    DeepTeamScanAdapter,
    deepteam_available,
)


def test_deepteam_available_is_bool():
    assert isinstance(deepteam_available(), bool)


@pytest.mark.skipif(not deepteam_available(), reason="deepteam not installed")
async def test_unexpected_kwarg_rejected():
    with pytest.raises(TypeError, match="unexpected keyword"):
        await DeepTeamScanAdapter().run(
            target=lambda x: x, description="d", probe="typo"
        )


@pytest.mark.skipif(not deepteam_available(), reason="deepteam not installed")
async def test_singleturn_with_only_multiturn_attacks_returns_empty():
    from giskard.checks import SuiteResult

    result = await DeepTeamScanAdapter().run(
        target=lambda x: x,
        description="d",
        attacks=["LinearJailbreaking"],  # multi-turn only
        target_mode="singleturn",  # -> filtered out -> empty
    )
    assert isinstance(result, SuiteResult)
    assert result.results == []
