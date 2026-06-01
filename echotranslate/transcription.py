"""Speech-to-text with OpenAI Whisper, for live mode.

Whisper transcribes captured audio and detects its language so the live loop can
translate it. The library is imported lazily inside :meth:`SpeechTranscriber.load`
so the package installs and tests without it.

Audio is passed to Whisper as an in-memory float32 array, so no system ffmpeg is
required (ffmpeg is only needed when Whisper loads audio from a file path).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from echotranslate.config import Settings
from echotranslate.errors import HeavyDependencyError, ModelNotAvailableError

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class TranscriptionResult:
    """The outcome of transcribing one audio segment.

    Attributes:
        text: The recognised text, stripped of surrounding whitespace.
        language: The detected source-language code (e.g. ``"en"``).
    """

    text: str
    language: str


class SpeechTranscriber:
    """Lazy wrapper around a Whisper model.

    Constructing this does not import or load anything; call :meth:`load` first.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded into memory."""
        return self._model is not None

    def load(self, *, progress: ProgressCallback | None = None) -> None:
        """Import Whisper and load the configured model.

        Args:
            progress: Optional callback for a "loading..." status message.

        Raises:
            HeavyDependencyError: If the ``voice`` extra is not installed.
            ModelNotAvailableError: If the model weights cannot be found or downloaded.
        """
        if self._model is not None:
            return
        try:
            import whisper
        except ImportError as exc:
            raise HeavyDependencyError(
                "Live mode needs the 'voice' extra (which includes Whisper):\n"
                "    pip install 'echotranslate[voice]'"
            ) from exc

        if progress is not None:
            progress("Loading speech recognition model...")
        try:
            self._model = whisper.load_model(
                self._settings.whisper_model,
                download_root=str(self._settings.whisper_cache_dir),
            )
        except (RuntimeError, FileNotFoundError) as exc:
            raise ModelNotAvailableError(
                f"Could not load the Whisper '{self._settings.whisper_model}' "
                "model. Download it while online with:\n"
                "    python download_whisper.py"
            ) from exc

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe a mono float32 audio array.

        Args:
            audio: Samples at 16 kHz (Whisper's expected rate).

        Returns:
            The recognised text and detected language.

        Raises:
            ModelNotAvailableError: If :meth:`load` has not been called.
        """
        if self._model is None:
            raise ModelNotAvailableError("Call load() before transcribing.")
        result = self._model.transcribe(audio, fp16=False)
        text = str(result.get("text", "")).strip()
        language = str(result.get("language", "en"))
        return TranscriptionResult(text=text, language=language)
