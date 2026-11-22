"""NeuroScope Python implementation of the calcMUA method.

The scientific method was developed by Dr. Cecilia Herbert. Fiamma Liz Leites
independently implemented it in MATLAB as ``calcmua_verFi.m``; this module is a
separate Python implementation based on that MATLAB implementation. This
provenance does not imply that Dr. Herbert authored or endorses NeuroScope AI.
Numerical parity with MATLAB reference outputs has not been established.

This module accepts an already-prepared continuous signal. It does not filter,
resample, normalize, or otherwise modify that signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sqrt
from numbers import Integral, Real
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

NOISE_DIVISOR = 0.6745
PROMINENCE_FACTOR = 0.75
AMPLITUDE_CAP_FACTOR = 25.0
IMPLEMENTATION_VERSION = "calcMUA-matlab-mapping-v1"


@dataclass(frozen=True, slots=True)
class CalcMUAResult:
    """Numerical results with zero-based sample indices and explicit time grid.

    ``mua`` is ``(samples, channels)``. ``evaluation_times_s`` follows MATLAB's
    one-based ``(1:N)/sampling_rate`` grid. Event indices are zero-based relative
    to the supplied signal; event times follow the source's ``index/Fs`` formula.
    Arrays are marked read-only before returning.
    """

    mua: npt.NDArray[np.float64]
    evaluation_times_s: npt.NDArray[np.float64]
    times_so: npt.NDArray[np.float64]
    thresholds: npt.NDArray[np.float64]
    noise_estimates: npt.NDArray[np.float64]
    spike_sample_indices: tuple[npt.NDArray[np.int64], ...]
    spike_times_s: tuple[npt.NDArray[np.float64], ...]
    spike_amplitudes: tuple[npt.NDArray[np.float64], ...]
    mode: Literal["rel", "abs"]


def _readonly(array: npt.NDArray) -> npt.NDArray:
    array.setflags(write=False)
    return array


def _normal_kde_at_points(
    events_s: npt.NDArray[np.float64],
    evaluation_times_s: npt.NDArray[np.float64],
    bandwidth_s: float,
) -> npt.NDArray[np.float64]:
    """Evaluate MATLAB's default unbounded normal-kernel PDF in bounded blocks."""
    if events_s.size == 0:
        return np.full(evaluation_times_s.shape, np.nan, dtype=np.float64)

    result = np.empty(evaluation_times_s.shape, dtype=np.float64)
    scale = 1.0 / (events_s.size * bandwidth_s * sqrt(2.0 * pi))
    # Bound the temporary pairwise matrix to roughly one million float values.
    event_block = min(events_s.size, 2048)
    point_block = max(1, 1_000_000 // event_block)
    for point_start in range(0, evaluation_times_s.size, point_block):
        point_stop = min(point_start + point_block, evaluation_times_s.size)
        points = evaluation_times_s[point_start:point_stop]
        density_sum = np.zeros(points.shape, dtype=np.float64)
        for event_start in range(0, events_s.size, event_block):
            event_stop = min(event_start + event_block, events_s.size)
            standardized = (
                points[:, None] - events_s[None, event_start:event_stop]
            ) / bandwidth_s
            density_sum += np.exp(-0.5 * standardized * standardized).sum(axis=1)
        result[point_start:point_stop] = density_sum * scale
    return result


def calc_mua(
    filtered_signal: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
    stdmin: float,
    mua_window_s: float,
    mode: Literal["rel", "abs"],
    num_points: int,
) -> CalcMUAResult:
    """Calculate the original per-channel thresholded KDE MUA trace.

    Input arrays may be one-dimensional or two-dimensional. A 2D array follows
    the MATLAB source convention: rows are samples and columns are channels.
    MATLAB peak locations are translated to zero-based indices in the result.
    The selected interval's first and final samples remain excluded from peak
    detection by ``find_peaks`` and do not require signal context beyond it.

    The explicit sample-time evaluation vector makes ``num_points`` redundant
    for output length, as in the source call. It is still required, validated,
    and preserved by callers for source-parameter provenance.
    """
    values = np.asarray(filtered_signal)
    if values.ndim not in (1, 2):
        raise ValueError("filtered_signal must be a non-empty 1D or 2D array.")
    if values.size == 0 or any(size == 0 for size in values.shape):
        raise ValueError("filtered_signal must not be empty.")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise TypeError("filtered_signal must contain real numeric values.")
    if not np.isfinite(values).all():
        raise ValueError("filtered_signal must contain only finite values.")
    if values.ndim == 1:
        samples_by_channels = values[:, None]
    else:
        samples_by_channels = values

    for name, value in (
        ("sampling_rate_hz", sampling_rate_hz),
        ("stdmin", stdmin),
        ("mua_window_s", mua_window_s),
    ):
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
            raise ValueError(f"{name} must be a finite real number.")
    sampling_rate_hz = float(sampling_rate_hz)
    stdmin = float(stdmin)
    mua_window_s = float(mua_window_s)
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be greater than zero.")
    if mua_window_s <= 0:
        raise ValueError("mua_window_s must be greater than zero.")
    if isinstance(num_points, bool) or not isinstance(num_points, Integral) or num_points <= 0:
        raise ValueError("num_points must be a positive integer.")
    if mode not in ("rel", "abs"):
        raise ValueError("mode must be exactly 'rel' or 'abs'.")

    sample_count, channel_count = samples_by_channels.shape
    evaluation_times = np.arange(1, sample_count + 1, dtype=np.float64) / sampling_rate_hz
    mua = np.empty((sample_count, channel_count), dtype=np.float64)
    times_so = np.empty((sample_count, channel_count), dtype=np.float64)
    thresholds = np.empty(channel_count, dtype=np.float64)
    noise_estimates = np.empty(channel_count, dtype=np.float64)
    all_indices: list[npt.NDArray[np.int64]] = []
    all_times: list[npt.NDArray[np.float64]] = []
    all_amplitudes: list[npt.NDArray[np.float64]] = []

    for channel_index in range(channel_count):
        signal = np.asarray(samples_by_channels[:, channel_index], dtype=np.float64)
        noise = float(np.median(np.abs(signal)) / NOISE_DIVISOR)
        threshold = stdmin * noise
        if not isfinite(noise) or not isfinite(threshold):
            raise ValueError("calcMUA noise or threshold overflowed for the supplied finite signal and parameters.")
        thresholds[channel_index] = threshold
        noise_estimates[channel_index] = noise

        if threshold < 0:
            peak_signal = -signal
        else:
            peak_signal = signal
        peak_height = abs(threshold)
        peak_prominence = PROMINENCE_FACTOR * abs(threshold)
        peak_indices, properties = find_peaks(
            peak_signal,
            height=peak_height,
            prominence=peak_prominence,
            plateau_size=(None, None),
        )
        if peak_indices.size:
            # MATLAB findpeaks reports the lowest-index sample of a flat peak;
            # SciPy reports its middle sample. Keep SciPy's measured properties
            # but translate plateau locations to the source convention.
            peak_indices = properties["left_edges"].astype(np.int64, copy=False)
            neural_peak_values = peak_signal[peak_indices]
            cap = AMPLITUDE_CAP_FACTOR * noise
            keep = (neural_peak_values < cap) & (neural_peak_values > -cap)
            peak_indices = peak_indices[keep]
        else:
            peak_indices = np.empty(0, dtype=np.int64)

        signed_amplitudes = signal[peak_indices].astype(np.float64, copy=True)
        # Source spike time = MATLAB one-based location / Fs.
        spike_times = (peak_indices.astype(np.float64) + 1.0) / sampling_rate_hz
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            density = _normal_kde_at_points(spike_times, evaluation_times, mua_window_s)
        if mode == "abs" and spike_times.size:
            density *= spike_times.size
        if spike_times.size and not np.isfinite(density).all():
            raise ValueError("calcMUA KDE produced non-finite values for the supplied bandwidth.")
        mua[:, channel_index] = density
        times_so[:, channel_index] = (
            evaluation_times if spike_times.size else np.nan
        )
        all_indices.append(_readonly(peak_indices.astype(np.int64, copy=True)))
        all_times.append(_readonly(spike_times))
        all_amplitudes.append(_readonly(signed_amplitudes))

    return CalcMUAResult(
        mua=_readonly(mua),
        evaluation_times_s=_readonly(evaluation_times),
        times_so=_readonly(times_so),
        thresholds=_readonly(thresholds),
        noise_estimates=_readonly(noise_estimates),
        spike_sample_indices=tuple(all_indices),
        spike_times_s=tuple(all_times),
        spike_amplitudes=tuple(all_amplitudes),
        mode=mode,
    )
