"""Tests for the CLI entry point and its offline-first startup ordering."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import echotranslate
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
    assert cli.main([]) == 0
    tui_cls.return_value.run.assert_called_once_with()
    # The offline package check must happen before the menu opens.
    assert calls == ["missing", "run"]


def test_main_keyboard_interrupt_exits_zero(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tty: None
) -> None:
    tui_cls, _calls = _patch_runtime(monkeypatch, settings)
    tui_cls.return_value.run.side_effect = KeyboardInterrupt
    assert cli.main([]) == 0


def test_main_non_tty_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    tui_cls = MagicMock()
    monkeypatch.setattr(cli, "TUI", tui_cls)
    assert cli.main([]) == 1
    tui_cls.assert_not_called()


def test_help_works_without_a_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # --help used to hit the TTY check first and print only an error.
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage: echotranslate" in out
    assert "./voices/" in out


def test_version_flag_reports_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert (
        capsys.readouterr().out.strip() == f"echotranslate {echotranslate.__version__}"
    )


def test_unknown_argument_exits_two_before_opening_the_menu(
    monkeypatch: pytest.MonkeyPatch, tty: None
) -> None:
    tui_cls = MagicMock()
    monkeypatch.setattr(cli, "TUI", tui_cls)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--bogus"])
    assert exit_info.value.code == 2
    tui_cls.assert_not_called()
