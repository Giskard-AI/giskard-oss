"""PEP 561 marker is packaged with giskard.scan."""

from importlib.resources import files


def test_pep561_py_typed_marker_is_present():
    marker = files("giskard.scan").joinpath("py.typed")
    assert marker.is_file()
