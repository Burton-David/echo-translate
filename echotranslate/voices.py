"""Voice-profile discovery.

Voice profiles are plain WAV files in the voices directory; the filename stem is
the profile name. This module is the single place that knows that convention.
"""

from __future__ import annotations

from pathlib import Path

from echotranslate.config import Settings


def list_voices(settings: Settings) -> list[str]:
    """Return the sorted names of saved voice profiles.

    A missing voices directory is a normal "nothing recorded yet" state and
    returns an empty list; an unexpected filesystem error (for example a
    permissions problem on an existing directory) propagates to the caller.

    Args:
        settings: Resolved settings providing ``voices_dir``.

    Returns:
        Profile names (filename stems) sorted alphabetically.
    """
    if not settings.voices_dir.is_dir():
        return []
    return sorted(path.stem for path in settings.voices_dir.glob("*.wav"))


def voice_path(settings: Settings, name: str) -> Path:
    """Return the WAV path for a voice profile name."""
    return settings.voices_dir / f"{name}.wav"


def voice_exists(settings: Settings, name: str) -> bool:
    """Return whether a voice profile WAV exists for ``name``."""
    return voice_path(settings, name).is_file()
