"""Pitch-contour analysis for pronunciation practice.

Extracts the fundamental-frequency (F0) track of a clip, compares two tracks by
their *shape* rather than their absolute pitch (so a deep voice and a high voice
are judged on equal footing), and renders the two contours as an overlaid ASCII
chart for the terminal.

This is what powers the tone/pitch feedback: hear a phrase in your own voice, say
it back, and see where your pitch follows the target and where it drifts. Only
NumPy is required, so the analysis runs and is tested without any of the
machine-learning extras.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 70-400 Hz spans ordinary speech; going wider invites octave errors.
_FMIN = 70.0
_FMAX = 400.0
# Average per-step pitch deviation (in semitones) that maps to a zero match score.
_SCORE_FLOOR_SEMITONES = 6.0


@dataclass(frozen=True, eq=False)
class PitchContour:
    """A fundamental-frequency track over time.

    Attributes:
        times: Frame-centre times in seconds.
        f0: Frequency per frame in hertz; ``NaN`` marks an unvoiced frame.
        sample_rate: Sample rate the contour was extracted at.
    """

    times: np.ndarray
    f0: np.ndarray
    sample_rate: int

    @property
    def voiced_f0(self) -> np.ndarray:
        voiced: np.ndarray = self.f0[~np.isnan(self.f0)]
        return voiced

    @property
    def has_pitch(self) -> bool:
        # Two voiced frames is the floor for a comparable contour.
        return int(np.count_nonzero(~np.isnan(self.f0))) >= 2


@dataclass(frozen=True, eq=False)
class ComparisonResult:
    """The outcome of comparing two pitch contours.

    Attributes:
        match: A heuristic 0-100 score for how closely the attempt's pitch shape
            follows the target's. Higher is closer. This is a pitch-contour
            similarity, not a full linguistic pronunciation grade.
        enough_data: Whether both contours had enough voiced audio to compare.
        message: A short explanation shown when ``enough_data`` is False.
    """

    match: float
    enough_data: bool
    message: str = ""


def extract_f0(
    samples: np.ndarray,
    sample_rate: int,
    *,
    fmin: float = _FMIN,
    fmax: float = _FMAX,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
    voicing_threshold: float = 0.3,
) -> PitchContour:
    """Estimate the F0 contour of an audio clip by short-time autocorrelation.

    Args:
        samples: Mono (or multi-channel, averaged) audio samples.
        sample_rate: Sample rate of ``samples`` in hertz.
        fmin: Lowest F0 to consider, in hertz.
        fmax: Highest F0 to consider, in hertz.
        frame_ms: Analysis window length in milliseconds.
        hop_ms: Step between analysis windows in milliseconds.
        voicing_threshold: Minimum normalised autocorrelation peak for a frame to
            count as voiced.

    Returns:
        A :class:`PitchContour`. Frames that are too quiet or not periodic enough
        are marked unvoiced (``NaN``).
    """
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.reshape(-1)

    frame_len = int(round(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    empty = PitchContour(np.empty(0), np.empty(0), int(sample_rate))
    if frame_len < 2 or audio.size < frame_len:
        return empty

    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(frame_len - 1, int(sample_rate / fmin))
    if max_lag <= min_lag:
        return empty

    window = np.hanning(frame_len)
    n_frames = 1 + (audio.size - frame_len) // hop
    starts = np.arange(n_frames) * hop
    frames = np.stack([audio[s : s + frame_len] * window for s in starts])
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    reference_rms = float(rms.max()) if rms.size else 0.0
    energy_gate = 0.15 * reference_rms

    times = (starts + frame_len / 2.0) / sample_rate
    f0 = np.full(n_frames, np.nan)
    # Autocorrelation rather than cepstrum: steadier F0 on short, noisy clips
    # from a laptop or phone mic.
    for i in range(n_frames):
        if reference_rms <= 0.0 or rms[i] < energy_gate:
            continue
        autocorr = _autocorrelation(frames[i])
        if autocorr[0] <= 0.0:
            continue
        search = autocorr[min_lag : max_lag + 1]
        if search.size == 0:
            continue
        peak_lag = int(np.argmax(search)) + min_lag
        if autocorr[peak_lag] / autocorr[0] < voicing_threshold:
            continue
        refined_lag = _parabolic_peak(autocorr, peak_lag)
        frequency = sample_rate / refined_lag
        if fmin <= frequency <= fmax:
            f0[i] = frequency

    return PitchContour(times, f0, int(sample_rate))


def compare_contours(target: PitchContour, attempt: PitchContour) -> ComparisonResult:
    """Compare two contours by pitch *shape*, ignoring absolute register.

    Both contours are converted to semitones relative to their own median pitch
    (which removes the speaker's baseline pitch), then aligned with dynamic time
    warping so differences in timing and tempo do not penalise the score.

    Args:
        target: The reference contour (the clip spoken in the user's voice).
        attempt: The user's recorded attempt.

    Returns:
        A :class:`ComparisonResult` with a 0-100 match score, or ``enough_data``
        set to False when either clip lacks voiced audio.
    """
    target_shape = _shape_semitones(target)
    attempt_shape = _shape_semitones(attempt)
    if target_shape.size < 2 or attempt_shape.size < 2:
        return ComparisonResult(
            0.0,
            False,
            "Not enough voiced audio to compare. Speak a little longer and louder.",
        )

    mean_cost = _dtw_mean_cost(target_shape, attempt_shape)
    match = 100.0 * max(0.0, 1.0 - mean_cost / _SCORE_FLOOR_SEMITONES)
    return ComparisonResult(round(match, 1), True)


def render_contours(
    target: PitchContour,
    attempt: PitchContour,
    *,
    width: int = 56,
    height: int = 11,
) -> list[str]:
    """Render the two contours as an overlaid ASCII chart.

    The top row is the highest pitch. ``#`` is the target, ``o`` is the attempt,
    and ``*`` is where they coincide. Each contour is resampled to ``width``
    columns and normalised to semitones relative to its own median, so the chart
    shows pitch *movement* rather than absolute height.

    Returns:
        ``height`` strings of length ``width``, or an empty list when either
        contour lacks enough voiced audio to plot.
    """
    target_shape = _shape_semitones(target)
    attempt_shape = _shape_semitones(attempt)
    if target_shape.size < 2 or attempt_shape.size < 2:
        return []

    target_row = _resample(target_shape, width)
    attempt_row = _resample(attempt_shape, width)
    low = float(min(target_row.min(), attempt_row.min()))
    high = float(max(target_row.max(), attempt_row.max()))
    if high - low < 1e-6:
        high = low + 1.0

    def to_row(value: float) -> int:
        fraction = (value - low) / (high - low)
        row = int(round((1.0 - fraction) * (height - 1)))
        return min(height - 1, max(0, row))

    grid = [[" "] * width for _ in range(height)]
    for column in range(width):
        t_row = to_row(float(target_row[column]))
        a_row = to_row(float(attempt_row[column]))
        grid[t_row][column] = "#"
        grid[a_row][column] = "o" if grid[a_row][column] == " " else "*"
    return ["".join(row) for row in grid]


def _autocorrelation(frame: np.ndarray) -> np.ndarray:
    n = frame.size
    spectrum = np.fft.rfft(frame, 2 * n)
    return np.fft.irfft(spectrum * np.conjugate(spectrum), 2 * n)[:n]


def _parabolic_peak(values: np.ndarray, index: int) -> float:
    # Sub-sample interpolation around the integer lag, so F0 is not quantised to
    # the sample rate.
    if index <= 0 or index >= values.size - 1:
        return float(index)
    previous, here, following = values[index - 1], values[index], values[index + 1]
    denominator = previous - 2.0 * here + following
    if denominator == 0.0:
        return float(index)
    return float(index + 0.5 * (previous - following) / denominator)


def _shape_semitones(contour: PitchContour) -> np.ndarray:
    voiced = contour.voiced_f0
    if voiced.size < 2:
        return np.empty(0)
    # Semitones relative to the clip's own median, so a deep voice and a high one
    # line up on shape instead of absolute pitch.
    reference = float(np.median(voiced))
    return 12.0 * np.log2(voiced / reference)


def _dtw_mean_cost(a: np.ndarray, b: np.ndarray) -> float:
    n, m = a.size, b.size
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        a_value = a[i - 1]
        for j in range(1, m + 1):
            distance = abs(a_value - b[j - 1])
            cost[i, j] = distance + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )

    i, j, steps = n, m, 0
    while i > 0 or j > 0:
        steps += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            move = int(np.argmin((cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])))
            if move == 0:
                i, j = i - 1, j - 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
    return float(cost[n, m] / max(1, steps))


def _resample(values: np.ndarray, width: int) -> np.ndarray:
    if values.size == width:
        return values
    positions = np.linspace(0.0, values.size - 1, width)
    resampled: np.ndarray = np.interp(positions, np.arange(values.size), values)
    return resampled
