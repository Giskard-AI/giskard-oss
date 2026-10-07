import pytest
from giskard.core.utils import is_true_env_str


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("false", False),
        ("maybe", False),
        ("1", True),
        (" true ", True),
        ("YES", True),
    ],
)
def test_is_true_env_str(value, expected):
    assert is_true_env_str(value) is expected
