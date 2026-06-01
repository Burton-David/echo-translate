"""Tests around the heavy ML backends.

Two kinds live here:

* Light "gate" tests that run in the default suite. They verify that the lazy
  imports raise a clear, actionable error when the optional extra or the model
  weights are absent, without downloading anything.
* ``@pytest.mark.heavy`` tests that exercise the real XTTS/Whisper models. They
  are excluded from the default run and additionally skip themselves when the
  models are not installed, so they never fail for the wrong reason.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import pytest

from echotranslate import config
from echotranslate.config import RECORD_SAMPLE_RATE, Settings
from echotranslate.errors import HeavyDependencyError, ModelNotAvailableError
from echotranslate.languages import by_menu_number
from echotranslate.synthesis import VoiceSynthesizer
from echotranslate.transcription import SpeechTranscriber

_HAS_TTS = importlib.util.find_spec("TTS") is not None
_HAS_WHISPER = importlib.util.find_spec("faster_whisper") is not None


def test_synthesizer_missing_extra_is_actionable(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setitem(sys.modules, "TTS", None)
    with pytest.raises(HeavyDependencyError) as info:
        VoiceSynthesizer(settings).load()
    assert "echotranslate[voice]" in str(info.value)


def test_synthesizer_reports_missing_model(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    # Make the import succeed but the weights look absent.
    fake_api = types.ModuleType("TTS.api")
    fake_api.TTS = lambda *_a, **_k: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "TTS", types.ModuleType("TTS"))
    monkeypatch.setitem(sys.modules, "TTS.api", fake_api)
    monkeypatch.setattr(config, "is_xtts_model_present", lambda: False)
    with pytest.raises(ModelNotAvailableError):
        VoiceSynthesizer(settings).load()


def test_synthesize_before_load_raises(settings: Settings, tmp_path) -> None:
    synth = VoiceSynthesizer(settings)
    with pytest.raises(ModelNotAvailableError):
        synth.synthesize_to_file(
            "hola", tmp_path / "voice.wav", by_menu_number(2), tmp_path / "out.wav"
        )


def test_transcriber_missing_extra_is_actionable(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(HeavyDependencyError) as info:
        SpeechTranscriber(settings).load()
    assert "echotranslate[voice]" in str(info.value)


def test_transcribe_before_load_raises(settings: Settings) -> None:
    transcriber = SpeechTranscriber(settings)
    with pytest.raises(ModelNotAvailableError):
        transcriber.transcribe(np.zeros(16000, dtype=np.float32))


@pytest.mark.heavy
def test_real_xtts_synthesis(settings: Settings, tmp_path) -> None:
    if not _HAS_TTS or not config.is_xtts_model_present():
        pytest.skip("XTTS package or weights not installed")
    config.configure_environment()
    reference = tmp_path / "reference.wav"
    seconds = np.linspace(0, 3, RECORD_SAMPLE_RATE * 3, dtype=np.float32)
    tone = (0.2 * np.sin(2 * np.pi * 140 * seconds)).astype(np.float32)
    import soundfile as sf

    sf.write(str(reference), tone, RECORD_SAMPLE_RATE)

    synth = VoiceSynthesizer(settings)
    synth.load()
    output = tmp_path / "out.wav"
    synth.synthesize_to_file("Hola, buenos días.", reference, by_menu_number(2), output)
    assert output.is_file() and output.stat().st_size > 0


@pytest.mark.heavy
def test_real_whisper_transcription(settings: Settings) -> None:
    if not _HAS_WHISPER:
        pytest.skip("Whisper package not installed")
    transcriber = SpeechTranscriber(settings)
    transcriber.load()
    result = transcriber.transcribe(np.zeros(16000, dtype=np.float32))
    # Silence should transcribe without error and report a language code.
    assert isinstance(result.text, str)
    assert isinstance(result.language, str) and result.language
