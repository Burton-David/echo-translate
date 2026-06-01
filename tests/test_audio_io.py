"""Tests for audio file I/O, capture wiring, and speech segmentation.

No real microphone or speaker is used: file I/O goes through soundfile with
synthetic arrays, and the capture path is driven by a fake sounddevice stream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from echotranslate import audio
from echotranslate.errors import (
    AudioDeviceError,
    EmptyRecordingError,
    HeavyDependencyError,
)


def test_wav_round_trip(tmp_path: Path) -> None:
    samples = np.linspace(-0.5, 0.5, 2205, dtype=np.float32)
    path = tmp_path / "clip.wav"
    audio.write_wav(path, samples, 22050)
    read_back, sample_rate = audio.read_wav(path)
    assert sample_rate == 22050
    assert read_back.shape[0] == samples.shape[0]
    # 16-bit WAV quantisation introduces a small but bounded error.
    assert np.allclose(read_back, samples, atol=1e-3)


def test_read_wav_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AudioDeviceError):
        audio.read_wav(tmp_path / "nope.wav")


class _FakeStream:
    """A context-manager stand-in that feeds chunks to the callback on entry."""

    def __init__(self, callback, chunks):  # type: ignore[no-untyped-def]
        self._callback = callback
        self._chunks = chunks

    def __enter__(self) -> _FakeStream:
        for chunk in self._chunks:
            self._callback(chunk, len(chunk), None, None)
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


class _FakeSounddevice:
    PortAudioError = type("PortAudioError", (Exception,), {})

    def __init__(self, chunks):  # type: ignore[no-untyped-def]
        self._chunks = chunks

    def InputStream(self, **kwargs):  # type: ignore[no-untyped-def]  # noqa: N802
        return _FakeStream(kwargs["callback"], self._chunks)


def test_record_until_enter_collects_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        np.full((4, 1), 0.2, dtype=np.float32),
        np.full((4, 1), 0.4, dtype=np.float32),
    ]
    monkeypatch.setattr(audio, "_import_sounddevice", lambda: _FakeSounddevice(chunks))
    recording = audio.record_until_enter(22050, wait_for_stop=lambda: None)
    assert recording.shape == (8,)
    assert recording.dtype == np.float32


def test_record_until_enter_empty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio, "_import_sounddevice", lambda: _FakeSounddevice([]))
    with pytest.raises(EmptyRecordingError):
        audio.record_until_enter(22050, wait_for_stop=lambda: None)


def test_play_without_sounddevice_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(HeavyDependencyError):
        audio.play(np.zeros(10, dtype=np.float32), 16000)


def _loud(samples: int = 1600) -> np.ndarray:
    return np.full(samples, 0.5, dtype=np.float32)


def _silent(samples: int = 1600) -> np.ndarray:
    return np.zeros(samples, dtype=np.float32)


def test_detect_speech_segments_splits_on_silence() -> None:
    # 0.5s of speech, then 1.6s of silence (> 1.5s threshold) closes one segment.
    chunks = [_loud()] * 5 + [_silent()] * 16
    segments = list(
        audio.detect_speech_segments(chunks, sample_rate=16000, silence_seconds=1.5)
    )
    assert len(segments) == 1
    assert segments[0].size >= 5 * 1600


def test_detect_speech_segments_flushes_trailing_speech() -> None:
    chunks = [_silent()] * 3 + [_loud()] * 4
    segments = list(audio.detect_speech_segments(chunks, sample_rate=16000))
    assert len(segments) == 1


def test_detect_speech_segments_ignores_pure_silence() -> None:
    segments = list(audio.detect_speech_segments([_silent()] * 10, sample_rate=16000))
    assert segments == []


def test_detect_speech_segments_drops_brief_blips() -> None:
    # 0.1s of sound is below the 0.3s floor, so a cough should not emit a segment.
    chunks = [_loud()] + [_silent()] * 16
    segments = list(
        audio.detect_speech_segments(chunks, sample_rate=16000, min_speech_seconds=0.3)
    )
    assert segments == []


def test_detect_speech_segments_keeps_speech_above_floor() -> None:
    chunks = [_loud()] * 5 + [_silent()] * 16
    segments = list(
        audio.detect_speech_segments(chunks, sample_rate=16000, min_speech_seconds=0.3)
    )
    assert len(segments) == 1
