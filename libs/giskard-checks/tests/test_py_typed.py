"""PEP 561 marker is packaged with giskard.checks."""

from importlib.resources import files


def test_pep561_py_typed_marker_is_present():
    marker = files("giskard.checks").joinpath("py.typed")
    assert marker.is_file()
