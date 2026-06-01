"""Speech-to-text with faster-whisper, for live mode.

faster-whisper runs Whisper through CTranslate2, which is several times quicker
than the reference implementation and lets us load an int8-quantised model for
low-latency transcription on CPU (and float-precision on CUDA via
``device="auto"``). The library is imported lazily inside
:meth:`SpeechTranscriber.load` so the package installs and tests without it.

Audio is passed as an in-memory float32 array, and faster-whisper bundles its own
media decoding (PyAV), so no system ffmpeg is required.
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
    """Lazy wrapper around a faster-whisper model.

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
        """Import faster-whisper and load the configured model.

        Args:
            progress: Optional callback for a "loading..." status message.

        Raises:
            HeavyDependencyError: If the ``voice`` extra is not installed.
            ModelNotAvailableError: If the model weights cannot be found or
                downloaded.
        """
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise HeavyDependencyError(
                "Live mode needs the 'voice' extra (which includes faster-whisper):\n"
                "    pip install 'echotranslate[voice]'"
            ) from exc

        if progress is not None:
            progress("Loading speech recognition model...")
        try:
            self._model = WhisperModel(
                self._settings.whisper_model,
                device="auto",
                compute_type="int8",
                download_root=str(self._settings.whisper_cache_dir),
            )
        except (OSError, RuntimeError, ValueError) as exc:
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
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        segments, info = self._model.transcribe(samples, language=None)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptionResult(text=text, language=str(info.language))
