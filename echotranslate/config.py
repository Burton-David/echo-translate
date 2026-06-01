"""Paths, settings, and environment configuration.

This module is intentionally free of machine-learning imports. It knows where
voice profiles and rendered audio live, where the XTTS weights would be on disk,
and how to build a deterministic output filename. Everything here is testable
without a model or a microphone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Identifier the Coqui TTS API uses to locate/download the XTTS v2 weights.
XTTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

# Whisper model size used for live speech-to-text. "small" balances accuracy and
# footprint (~460 MB); see ``download_whisper.py`` for the other options.
WHISPER_DEFAULT_MODEL = "small"

# XTTS expects its speaker reference clip at 22.05 kHz; Whisper expects 16 kHz.
RECORD_SAMPLE_RATE = 22050
LIVE_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Settings:
    """Resolved filesystem locations for one run.

    Attributes:
        voices_dir: Directory holding ``<name>.wav`` voice profiles.
        output_dir: Directory holding rendered audio, organised by date.
        whisper_cache_dir: Where the Whisper model weights are cached.
        whisper_model: Whisper model size to load for live mode.
    """

    voices_dir: Path
    output_dir: Path
    whisper_cache_dir: Path
    whisper_model: str = WHISPER_DEFAULT_MODEL


def default_settings(base_dir: Path | None = None) -> Settings:
    """Build settings rooted at ``base_dir`` (the current directory if omitted).

    Args:
        base_dir: Root for ``voices/`` and ``output/``. Defaults to the process's
            current working directory.

    Returns:
        A :class:`Settings` instance. Directories are not created here; call
        :func:`ensure_dirs` for that.
    """
    root = base_dir if base_dir is not None else Path.cwd()
    return Settings(
        voices_dir=root / "voices",
        output_dir=root / "output",
        whisper_cache_dir=Path.home() / ".cache" / "whisper",
    )


def ensure_dirs(settings: Settings) -> None:
    """Create the voices and output directories if they do not already exist."""
    settings.voices_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)


def configure_environment() -> None:
    """Set the environment variables the ML backends expect, without clobbering.

    Sets ``COQUI_TOS_AGREED`` (accepts the XTTS license non-interactively) and
    ``KMP_DUPLICATE_LIB_OK`` (avoids an OpenMP abort some PyTorch builds hit on
    macOS) only when they are not already set, so a user's explicit choice wins.
    """
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def xtts_model_locations() -> list[Path]:
    """Return the candidate on-disk locations for the downloaded XTTS weights."""
    model_dir = "tts_models--multilingual--multi-dataset--xtts_v2"
    home = Path.home()
    return [
        home / ".local" / "share" / "tts" / model_dir,
        home / "Library" / "Application Support" / "tts" / model_dir,
    ]


def is_xtts_model_present() -> bool:
    """Return whether the XTTS weights exist in any known cache location."""
    return any(location.exists() for location in xtts_model_locations())


def build_output_path(
    settings: Settings,
    voice: str,
    lang_code: str,
    now: datetime | None = None,
) -> Path:
    """Build the path for a rendered clip: ``output/<date>/<time>_<voice>_<lang>.wav``.

    Args:
        settings: Resolved settings providing ``output_dir``.
        voice: Voice profile name (used verbatim in the filename).
        lang_code: Target language code (used verbatim in the filename).
        now: Timestamp to use; defaults to :meth:`datetime.now`. Injectable so the
            path is deterministic in tests.

    Returns:
        The full output path. The date subdirectory is not created here.
    """
    moment = now if now is not None else datetime.now()
    date_dir = settings.output_dir / moment.strftime("%Y-%m-%d")
    filename = f"{moment.strftime('%H%M%S')}_{voice}_{lang_code}.wav"
    return date_dir / filename
