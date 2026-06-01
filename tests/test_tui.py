"""Tests for the terminal UI dispatch and guard logic.

The console is mocked, so these tests exercise control flow (which screen runs,
how errors are rendered, that builtin ``input`` is never used) without rendering
anything or reading real keystrokes.
"""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock

import pytest

from echotranslate import tui as tui_module
from echotranslate.config import Settings
from echotranslate.errors import HeavyDependencyError
from echotranslate.tui import TUI


def test_dispatch_exit_returns_false(console: MagicMock, settings: Settings) -> None:
    ui = TUI(console, settings)
    assert ui._dispatch("5") is False
    assert any(
        call.args and "Goodbye" in str(call.args[0])
        for call in console.print.call_args_list
    )


@pytest.mark.parametrize(
    ("choice", "method"),
    [
        ("1", "screen_record_voice"),
        ("2", "screen_translate_text"),
        ("3", "screen_live_translation"),
        ("4", "screen_list_translations"),
    ],
)
def test_dispatch_invokes_matching_screen(
    console: MagicMock,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
    method: str,
) -> None:
    ui = TUI(console, settings)
    screen = MagicMock()
    monkeypatch.setattr(ui, method, screen)
    assert ui._dispatch(choice) is True
    screen.assert_called_once_with()


def test_dispatch_renders_heavy_dependency_error(
    console: MagicMock, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    ui = TUI(console, settings)
    monkeypatch.setattr(
        ui,
        "screen_translate_text",
        MagicMock(
            side_effect=HeavyDependencyError("pip install 'echotranslate[voice]'")
        ),
    )
    assert ui._dispatch("2") is True
    printed = " ".join(
        str(call.args[0]) for call in console.print.call_args_list if call.args
    )
    assert "echotranslate[voice]" in printed


def test_translate_screen_without_voice_skips_synthesis(
    console: MagicMock, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tui_module, "list_voices", lambda _settings: [])
    ui = TUI(console, settings)
    ui.screen_translate_text()
    # No voice means we return before constructing or loading a synthesizer.
    assert ui._synthesizer is None


def test_menu_never_uses_builtin_input(
    console: MagicMock, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("builtin input() must not be used; use console.input")

    monkeypatch.setattr(builtins, "input", explode)
    monkeypatch.setattr(tui_module, "list_voices", lambda _settings: [])
    ui = TUI(console, settings)
    ui._render_menu()
    assert ui._dispatch("5") is False
