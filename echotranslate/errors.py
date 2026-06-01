"""Typed exceptions for EchoTranslate.

Each failure mode in the pipeline gets its own exception so the terminal UI can
render an actionable message (and tests can assert on a type rather than match a
string). The base class lets callers catch everything the package raises on
purpose while letting genuinely unexpected errors propagate with a real
traceback.
"""

from __future__ import annotations


class EchoTranslateError(Exception):
    """Base class for every error EchoTranslate raises deliberately."""


class HeavyDependencyError(EchoTranslateError):
    """An optional machine-learning extra is not installed.

    Raised when the ``voice`` extra (Coqui XTTS, OpenAI Whisper) or the ``audio``
    extra (sounddevice) is needed but its package cannot be imported. The message
    carries the exact ``pip install`` command that fixes it.
    """


class ModelNotAvailableError(EchoTranslateError):
    """A model's weights are not present on disk.

    The Python package is installed, but the multi-gigabyte XTTS weights or the
    Whisper model file have not been downloaded yet.
    """


class TranslationPackageError(EchoTranslateError):
    """A required Argos Translate language package is not installed.

    Raised when a language pair is needed but absent and downloading was not
    permitted (for example, because the caller asked to stay offline).
    """


class AudioDeviceError(EchoTranslateError):
    """A microphone or speaker could not be opened or used."""


class EmptyRecordingError(EchoTranslateError):
    """A recording finished without capturing any audio."""
