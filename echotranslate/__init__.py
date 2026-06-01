"""EchoTranslate: hear text spoken in your own cloned voice in another language.

A local, offline command-line tool for pronunciation practice. It translates
English text, then synthesizes it in a clone of your own voice so you can hear
exactly how a phrase should sound coming from you.

This top-level module deliberately imports nothing heavy. The machine-learning
backends (Coqui XTTS, OpenAI Whisper) are loaded lazily by
:mod:`echotranslate.synthesis` and :mod:`echotranslate.transcription` so that
``import echotranslate`` and the deterministic test suite run without the
voice-cloning model weights present.
"""

from __future__ import annotations

from echotranslate.errors import (
    AudioDeviceError,
    EchoTranslateError,
    EmptyRecordingError,
    HeavyDependencyError,
    ModelNotAvailableError,
    TranslationPackageError,
)

__version__ = "0.1.0"

__all__ = [
    "AudioDeviceError",
    "EchoTranslateError",
    "EmptyRecordingError",
    "HeavyDependencyError",
    "ModelNotAvailableError",
    "TranslationPackageError",
    "__version__",
]
