# type: ignore[attr-defined]
"""Inspect your AI models visually, find bugs, give feedback 🕵️‍♀️ 💬"""

# The following code block is made to remove pydantic 2.0 warnings, especially for MLFlow
# https://docs.python.org/3/library/warnings.html#temporarily-suppressing-warnings
from importlib import metadata as importlib_metadata

from . import demo
from .core.suite import Suite, SuiteInput
from .core.test_result import TestResult
from .datasets.base import Dataset
from .models.automodel import Model
from .models.model_explanation import explain_with_shap
from .registry.decorators import test
from .registry.giskard_test import GiskardTest
from .registry.slicing_function import SlicingFunction, slicing_function
from .registry.transformation_function import TransformationFunction, transformation_function
from .scanner import scan
from .utils.analytics_collector import analytics
from .utils.logging_utils import configure_logging

configure_logging()


def get_version() -> str:
    try:
        res = importlib_metadata.version(__name__)
        if res is None:
            return importlib_metadata.distribution(__name__).version
        return res
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__: str = get_version()

analytics.track("Initialized giskard library")

__all__ = [
    "Dataset",
    "test",
    "Model",
    "Suite",
    "slicing_function",
    "transformation_function",
    "SuiteInput",
    "SlicingFunction",
    "TransformationFunction",
    "scan",
    "explain_with_shap",
    "TestResult",
    "GiskardTest",
    "demo",
]
