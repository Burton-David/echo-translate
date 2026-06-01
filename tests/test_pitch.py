"""Tests for pitch-contour extraction, comparison, and rendering.

All signals are synthesised, so the analysis is verified deterministically without
a microphone or any speech model.
"""

from __future__ import annotations

import numpy as np

from echotranslate.pitch import compare_contours, extract_f0, render_contours

_SR = 22050


def _tone(freq: float, *, seconds: float = 0.5, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(_SR * seconds)) / _SR
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _glide(
    f_start: float, f_end: float, *, seconds: float = 0.6, amplitude: float = 0.5
) -> np.ndarray:
    n = int(_SR * seconds)
    t = np.arange(n) / _SR
    instantaneous = f_start + (f_end - f_start) * (t / seconds)
    phase = 2.0 * np.pi * np.cumsum(instantaneous) / _SR
    return (amplitude * np.sin(phase)).astype(np.float32)


def test_extract_f0_tracks_a_steady_tone() -> None:
    contour = extract_f0(_tone(200.0), _SR)
    assert contour.has_pitch
    assert 190.0 <= float(np.median(contour.voiced_f0)) <= 210.0


def test_extract_f0_on_silence_finds_no_pitch() -> None:
    contour = extract_f0(np.zeros(_SR, dtype=np.float32), _SR)
    assert not contour.has_pitch
    assert contour.voiced_f0.size == 0


def test_extract_f0_follows_a_rising_glide() -> None:
    voiced = extract_f0(_glide(150.0, 250.0), _SR).voiced_f0
    third = voiced.size // 3
    assert float(np.median(voiced[:third])) < float(np.median(voiced[-third:]))


def test_extract_f0_handles_short_input() -> None:
    contour = extract_f0(np.zeros(8, dtype=np.float32), _SR)
    assert not contour.has_pitch


def test_compare_identical_contour_scores_high() -> None:
    contour = extract_f0(_glide(180.0, 230.0), _SR)
    result = compare_contours(contour, contour)
    assert result.enough_data
    assert result.match > 90.0


def test_compare_is_register_invariant() -> None:
    # Same rising shape, very different absolute pitch (a deep voice vs a high one).
    target = extract_f0(_glide(180.0, 230.0), _SR)
    deep_attempt = extract_f0(_glide(110.0, 140.0), _SR)
    result = compare_contours(target, deep_attempt)
    assert result.enough_data
    assert result.match > 70.0


def test_compare_rising_against_falling_scores_low() -> None:
    rising = extract_f0(_glide(180.0, 230.0), _SR)
    falling = extract_f0(_glide(230.0, 180.0), _SR)
    identical = compare_contours(rising, rising).match
    opposite = compare_contours(rising, falling).match
    assert opposite < 70.0
    assert opposite < identical


def test_compare_without_voiced_audio_reports_insufficient() -> None:
    silence = extract_f0(np.zeros(_SR, dtype=np.float32), _SR)
    result = compare_contours(silence, silence)
    assert not result.enough_data
    assert result.match == 0.0
    assert result.message


def test_render_returns_a_fixed_grid() -> None:
    contour = extract_f0(_glide(180.0, 230.0), _SR)
    lines = render_contours(contour, contour, width=40, height=8)
    assert len(lines) == 8
    assert all(len(line) == 40 for line in lines)


def test_render_identical_contours_overlap() -> None:
    contour = extract_f0(_glide(180.0, 230.0), _SR)
    joined = "".join(render_contours(contour, contour))
    assert "*" in joined
    assert set(joined) <= {" ", "*"}


def test_render_different_shapes_show_both_markers() -> None:
    rising = extract_f0(_glide(180.0, 230.0), _SR)
    falling = extract_f0(_glide(230.0, 180.0), _SR)
    joined = "".join(render_contours(rising, falling))
    assert "#" in joined
    assert "o" in joined


def test_render_without_voiced_audio_is_empty() -> None:
    silence = extract_f0(np.zeros(_SR, dtype=np.float32), _SR)
    assert render_contours(silence, silence) == []
