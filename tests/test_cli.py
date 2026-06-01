"""Tests for the CLI entry point and its offline-first startup ordering."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from echotranslate import cli
from echotranslate.config import Settings


@pytest.fixture
def tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> tuple[MagicMock, list[str]]:
    """Stub the menu and settings; return the TUI mock and a call-order log."""
    calls: list[str] = []
    tui_cls = MagicMock()
    tui_cls.return_value.run.side_effect = lambda: calls.append("run")
    monkeypatch.setattr(cli, "TUI", tui_cls)
    monkeypatch.setattr(cli, "default_settings", lambda: settings)
    monkeypatch.setattr(
        cli, "missing_packages", lambda _targets: calls.append("missing") or []
    )
    return tui_cls, calls


def test_main_runs_menu_after_offline_check(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tty: None
) -> None:
    tui_cls, calls = _patch_runtime(monkeypatch, settings)
    assert cli.main() == 0
    tui_cls.return_value.run.assert_called_once_with()
    # The offline package check must happen before the menu opens.
    assert calls == ["missing", "run"]


def test_main_keyboard_interrupt_exits_zero(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tty: None
) -> None:
    tui_cls, _calls = _patch_runtime(monkeypatch, settings)
    tui_cls.return_value.run.side_effect = KeyboardInterrupt
    assert cli.main() == 0


def test_main_non_tty_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    tui_cls = MagicMock()
    monkeypatch.setattr(cli, "TUI", tui_cls)
    assert cli.main() == 1
    tui_cls.assert_not_called()
