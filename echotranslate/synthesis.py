"""Voice synthesis with Coqui XTTS v2.

XTTS clones a speaker from a short reference clip and reads target-language text
in that voice. The model and PyTorch are imported lazily inside
:meth:`VoiceSynthesizer.load` so the rest of the package installs and tests
without them.

Two failure modes are kept distinct: the ``voice`` extra not being installed
(:class:`HeavyDependencyError`) versus the package being present but the
multi-gigabyte weights not yet downloaded (:class:`ModelNotAvailableError`).

This uses the community-maintained ``coqui-tts`` fork, imported as
``from TTS.api import TTS``. Synthesis runs on CPU (the XTTS path does not
support Apple Silicon MPS), so expect a few seconds per sentence.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from echotranslate import config
from echotranslate.config import Settings
from echotranslate.errors import HeavyDependencyError, ModelNotAvailableError

if TYPE_CHECKING:
    from echotranslate.languages import Language

ProgressCallback = Callable[[str], None]


class VoiceSynthesizer:
    """Lazy wrapper around the XTTS model.

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
        """Import and load the XTTS model.

        Args:
            progress: Optional callback for a "loading..." status message.

        Raises:
            HeavyDependencyError: If the ``voice`` extra is not installed.
            ModelNotAvailableError: If the XTTS weights are not on disk.
        """
        if self._model is not None:
            return
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise HeavyDependencyError(
                "Voice synthesis needs the 'voice' extra:\n"
                "    pip install 'echotranslate[voice]'\n"
                "PyTorch may need to be installed separately; see the README."
            ) from exc

        if not config.is_xtts_model_present():
            raise ModelNotAvailableError(
                "The XTTS voice model (~2 GB) has not been downloaded yet. Run:\n"
                f'    python -c "from TTS.api import TTS; '
                f"TTS('{config.XTTS_MODEL_ID}')\""
            )

        if progress is not None:
            progress("Loading voice synthesis model...")
        # XTTS writes license and progress chatter to stderr while loading; keep
        # it contained to this call.
        with contextlib.redirect_stderr(io.StringIO()):
            self._model = TTS(config.XTTS_MODEL_ID, gpu=False)

    def synthesize_to_file(
        self,
        text: str,
        speaker_wav: Path,
        language: Language,
        output_path: Path,
    ) -> Path:
        """Render ``text`` in the speaker's voice and write it to ``output_path``.

        Args:
            text: Text to speak, already in the target language.
            speaker_wav: Reference recording that defines the cloned voice.
            language: Target language (its ``xtts_code`` is passed to the model).
            output_path: Destination WAV path; its parent is created if needed.

        Returns:
            ``output_path``.

        Raises:
            ModelNotAvailableError: If :meth:`load` has not been called.
        """
        if self._model is None:
            raise ModelNotAvailableError("Call load() before synthesizing.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stderr(io.StringIO()):
            self._model.tts_to_file(
                text=text,
                speaker_wav=str(speaker_wav),
                language=language.xtts_code,
                file_path=str(output_path),
            )
        return output_path
