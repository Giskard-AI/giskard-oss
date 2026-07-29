import inspect

from giskard.scan.quality import quality_scan
from giskard.scan.types import resolve_scan_options
from giskard.scan.vulnerability import vulnerability_scan

SHARED_SCAN_PARAMETERS = (
    "max_scenarios",
    "seed",
    "group_by",
    "parallel",
    "max_concurrency",
    "return_exception",
    "target_mode",
)

SHARED_SCAN_DEFAULTS = (
    "max_scenarios",
    "seed",
    "parallel",
    "max_concurrency",
    "return_exception",
    "target_mode",
)


def test_resolve_scan_options_matches_for_identical_shared_kwargs():
    quality_opts = resolve_scan_options(
        max_scenarios=20,
        seed=7,
        group_by="custom",
        parallel=False,
        max_concurrency=8,
        return_exception=True,
    )
    vulnerability_opts = resolve_scan_options(
        max_scenarios=20,
        seed=7,
        group_by="custom",
        parallel=False,
        max_concurrency=8,
        return_exception=True,
        commercial_use=False,
    )

    assert quality_opts == vulnerability_opts


def test_scan_entrypoints_expose_shared_explicit_kwargs():
    quality_params = inspect.signature(quality_scan).parameters
    vulnerability_params = inspect.signature(vulnerability_scan).parameters

    for name in SHARED_SCAN_PARAMETERS:
        assert name in quality_params
        assert name in vulnerability_params

    for name in SHARED_SCAN_DEFAULTS:
        assert quality_params[name].default == vulnerability_params[name].default

    assert quality_params["group_by"].default == "component"
    assert vulnerability_params["group_by"].default == "threat-type"

    assert "knowledge_base" in quality_params
    assert "commercial_use" in vulnerability_params
    assert "knowledge_base" not in vulnerability_params
    assert "commercial_use" not in quality_params
