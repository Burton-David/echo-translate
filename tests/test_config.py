"""Tests for configuration, paths, and environment setup."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

from echotranslate import config
from echotranslate.config import (
    Settings,
    build_output_path,
    configure_environment,
    default_settings,
    ensure_dirs,
)


def test_build_output_path_is_deterministic(settings: Settings) -> None:
    moment = datetime(2026, 6, 1, 12, 0, 0)
    path = build_output_path(settings, "alex", "es", now=moment)
    assert path == settings.output_dir / "2026-06-01" / "120000_alex_es.wav"


def test_configure_environment_sets_unset_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COQUI_TOS_AGREED", raising=False)
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    configure_environment()
    import os

    assert os.environ["COQUI_TOS_AGREED"] == "1"
    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"


def test_configure_environment_does_not_clobber(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COQUI_TOS_AGREED", "0")
    configure_environment()
    import os

    assert os.environ["COQUI_TOS_AGREED"] == "0"


def test_configure_environment_leaves_stderr_untouched() -> None:
    before = sys.stderr
    configure_environment()
    assert sys.stderr is before


def test_is_xtts_model_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    candidate = tmp_path / "xtts"
    monkeypatch.setattr(config, "xtts_model_locations", lambda: [candidate])
    assert config.is_xtts_model_present() is False
    candidate.mkdir()
    assert config.is_xtts_model_present() is True


def test_default_settings_roots_under_base(tmp_path: Path) -> None:
    s = default_settings(tmp_path)
    assert s.voices_dir == tmp_path / "voices"
    assert s.output_dir == tmp_path / "output"


def test_ensure_dirs_is_idempotent(settings: Settings) -> None:
    ensure_dirs(settings)
    ensure_dirs(settings)  # second call must not raise
    assert settings.voices_dir.is_dir()
    assert settings.output_dir.is_dir()
