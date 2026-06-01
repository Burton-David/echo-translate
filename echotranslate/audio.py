"""Audio capture, playback, and file I/O.

The file-I/O helpers (:func:`read_wav`, :func:`write_wav`) use soundfile and need
no hardware, so they are fully unit-testable. Capture and playback need a real
device and the optional ``sounddevice`` package (which in turn needs the
PortAudio system library), so that import is deferred until it is actually used.

The voice-activity logic in :func:`detect_speech_segments` is a pure generator
over audio chunks, separated from the live device loop so it can be tested with
synthetic input.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from echotranslate.errors import (
    AudioDeviceError,
    EmptyRecordingError,
    HeavyDependencyError,
)


def _import_sounddevice() -> Any:
    """Import sounddevice lazily, with an actionable error if it is absent.

    Returns the ``sounddevice`` module (typed ``Any`` because it ships no stubs).
    """
    try:
        import sounddevice as sd
    except OSError as exc:
        # sounddevice raises OSError when the PortAudio shared library is missing.
        raise HeavyDependencyError(
            "Audio capture/playback needs the PortAudio system library.\n"
            "    macOS:        brew install portaudio\n"
            "    Debian/Ubuntu: sudo apt install libportaudio2"
        ) from exc
    except ImportError as exc:
        raise HeavyDependencyError(
            "Audio capture/playback needs the 'audio' extra:\n"
            "    pip install 'echotranslate[audio]'"
        ) from exc
    return sd


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write a mono float32 sample array to ``path`` as a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file.

    Returns:
        A ``(samples, sample_rate)`` tuple.

    Raises:
        AudioDeviceError: If the file cannot be read.
    """
    try:
        samples, sample_rate = sf.read(str(path))
    except (RuntimeError, sf.SoundFileError) as exc:
        raise AudioDeviceError(f"Could not read audio file {path}: {exc}") from exc
    return samples, int(sample_rate)


def play(samples: np.ndarray, sample_rate: int, *, gain: float = 1.0) -> None:
    """Play a sample array through the default output device and block until done.

    Args:
        samples: Audio samples to play.
        sample_rate: Sample rate of ``samples`` in hertz.
        gain: Linear volume multiplier; values below 1.0 reduce live-mode
            feedback.

    Raises:
        HeavyDependencyError: If sounddevice/PortAudio is unavailable.
        AudioDeviceError: If playback fails on the device.
    """
    sd = _import_sounddevice()
    try:
        sd.play(samples * gain, sample_rate)
        sd.wait()
    except sd.PortAudioError as exc:
        raise AudioDeviceError(f"Playback failed: {exc}") from exc


def record_until_enter(
    sample_rate: int,
    *,
    wait_for_stop: Callable[[], object],
    on_tick: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Record from the microphone until ``wait_for_stop`` returns.

    Capture runs in sounddevice's callback thread; the caller's ``wait_for_stop``
    blocks the main thread (typically waiting for the user to press Enter). While
    recording, ``on_tick`` is called roughly ten times a second with the elapsed
    seconds so the UI can show a timer.

    Args:
        sample_rate: Capture rate in hertz.
        wait_for_stop: Blocking callable that returns when recording should stop.
        on_tick: Optional callback receiving elapsed seconds.

    Returns:
        A mono float32 array of the captured audio.

    Raises:
        HeavyDependencyError: If sounddevice/PortAudio is unavailable.
        AudioDeviceError: If the input stream cannot be opened.
        EmptyRecordingError: If no audio was captured.
    """
    sd = _import_sounddevice()
    chunks: list[np.ndarray] = []

    def callback(indata, _frames, _time_info, _status):  # type: ignore[no-untyped-def]
        chunks.append(indata.copy())

    stop = threading.Event()

    def tick() -> None:
        start = time.monotonic()
        while not stop.is_set():
            if on_tick is not None:
                on_tick(time.monotonic() - start)
            time.sleep(0.1)

    try:
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        )
    except sd.PortAudioError as exc:
        raise AudioDeviceError(f"Could not open the microphone: {exc}") from exc

    ticker = threading.Thread(target=tick, daemon=True)
    with stream:
        ticker.start()
        try:
            wait_for_stop()
        finally:
            stop.set()
    ticker.join()

    if not chunks:
        raise EmptyRecordingError("No audio was captured.")
    return np.concatenate(chunks, axis=0).reshape(-1)


def microphone_chunks(
    sample_rate: int,
    *,
    chunk_frames: int,
) -> Iterator[np.ndarray]:
    """Yield mono float32 chunks from the microphone until the consumer stops.

    The input stream is opened for the lifetime of the iterator and closed when
    iteration ends (including when the consumer breaks out of the loop). Pair this
    with :func:`detect_speech_segments` to turn a live mic into spoken segments.

    Args:
        sample_rate: Capture rate in hertz.
        chunk_frames: Frames per delivered chunk (e.g. 100 ms worth).

    Yields:
        Mono float32 sample arrays.

    Raises:
        HeavyDependencyError: If sounddevice/PortAudio is unavailable.
        AudioDeviceError: If the input stream cannot be opened.
    """
    sd = _import_sounddevice()
    pending: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, _frames, _time_info, _status):  # type: ignore[no-untyped-def]
        pending.put(indata.copy())

    try:
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
            callback=callback,
        )
    except sd.PortAudioError as exc:
        raise AudioDeviceError(f"Could not open the microphone: {exc}") from exc

    with stream:
        while True:
            try:
                yield pending.get(timeout=0.1).reshape(-1)
            except queue.Empty:
                continue


def detect_speech_segments(
    chunks: Iterable[np.ndarray],
    *,
    sample_rate: int,
    volume_threshold: float = 0.01,
    silence_seconds: float = 1.5,
) -> Iterator[np.ndarray]:
    """Group a stream of audio chunks into spoken segments by silence.

    A segment starts when a chunk's loudness (RMS) crosses ``volume_threshold``
    and ends after ``silence_seconds`` of quiet. This is the pure form of the
    live-mode listening loop, with no device or queue involved.

    Args:
        chunks: Iterable of mono sample arrays (e.g. 100 ms each).
        sample_rate: Sample rate of the chunks in hertz.
        volume_threshold: RMS above which a chunk counts as speech.
        silence_seconds: Trailing quiet needed to close a segment.

    Yields:
        One concatenated float32 array per detected segment.
    """
    buffer: list[np.ndarray] = []
    silence = 0.0
    speaking = False

    for raw in chunks:
        chunk = np.asarray(raw, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            continue
        seconds = chunk.size / sample_rate
        rms = float(np.sqrt(np.mean(np.square(chunk))))

        if rms > volume_threshold:
            speaking = True
            silence = 0.0
            buffer.append(chunk)
        elif speaking:
            silence += seconds
            buffer.append(chunk)
            if silence >= silence_seconds:
                yield np.concatenate(buffer)
                buffer = []
                silence = 0.0
                speaking = False

    if speaking and buffer:
        yield np.concatenate(buffer)
