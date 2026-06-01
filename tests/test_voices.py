"""Tests for voice-profile discovery."""

from __future__ import annotations

from echotranslate.config import Settings
from echotranslate.voices import list_voices, voice_exists, voice_path


def test_list_voices_returns_empty_when_dir_absent(settings: Settings) -> None:
    assert not settings.voices_dir.exists()
    assert list_voices(settings) == []


def test_list_voices_sorted_and_ignores_non_wav(settings: Settings) -> None:
    settings.voices_dir.mkdir(parents=True)
    (settings.voices_dir / "bravo.wav").touch()
    (settings.voices_dir / "alpha.wav").touch()
    (settings.voices_dir / "notes.txt").touch()
    assert list_voices(settings) == ["alpha", "bravo"]


def test_voice_path_and_exists(settings: Settings) -> None:
    settings.voices_dir.mkdir(parents=True)
    assert voice_exists(settings, "alex") is False
    voice_path(settings, "alex").touch()
    assert voice_exists(settings, "alex") is True
    assert voice_path(settings, "alex") == settings.voices_dir / "alex.wav"
