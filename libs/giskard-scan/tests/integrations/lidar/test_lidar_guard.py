import pytest
from giskard.scan.integrations.lidar import _adapter


def test_require_lidar_raises_private_message(monkeypatch):
    monkeypatch.setattr(_adapter, "find_spec", lambda name: None)
    with pytest.raises(ImportError) as exc_info:
        _adapter._require_lidar()
    message = str(exc_info.value)
    assert "private" in message.lower()
    assert "git+https://github.com/Giskard-AI/lidar.git@v0.2.7" in message


def test_lidar_available_reflects_find_spec(monkeypatch):
    monkeypatch.setattr(_adapter, "find_spec", lambda name: object())
    assert _adapter.lidar_available() is True
    monkeypatch.setattr(_adapter, "find_spec", lambda name: None)
    assert _adapter.lidar_available() is False
