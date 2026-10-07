"""Tests for giskard.checks package initialization."""

import importlib

import giskard.checks
import giskard.core.welcome


def test_import_shows_welcome(monkeypatch):
    """Importing giskard.checks invokes the welcome hook."""
    calls: list[object] = []
    monkeypatch.setattr(
        giskard.core.welcome,
        "maybe_show_welcome",
        lambda: calls.append(None),
    )

    importlib.reload(giskard.checks)

    assert calls == [None]
