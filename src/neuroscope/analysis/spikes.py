from __future__ import annotations
import numpy as np


def estimate_noise_mad(signal: np.ndarray) -> float:
    """Robust Gaussian-equivalent noise scale using MAD around the median."""
    x = np.asarray(signal, dtype=float).ravel()
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("signal must be non-empty and finite")
    return float(np.median(np.abs(x - np.median(x))) / 0.6744897501960817)


def detect_negative_spikes(signal: np.ndarray, sampling_rate_hz: float,
                           threshold_sd: float = -4.0, *, channel: int | None = None,
                           channel_axis: int | None = None) -> dict:
    """Minimal threshold detector for extracellular negative-going events.

    This is a transparent MVP detector, not a substitute for refractory-period,
    waveform, artifact, or multi-channel quality-control procedures.
    """
    x = np.asarray(signal)
    if x.ndim > 1:
        if x.ndim != 2 or channel is None or channel_axis not in (0, 1):
            raise ValueError("Multidimensional input requires an explicit channel and channel_axis (0 or 1).")
        if channel < 0 or channel >= x.shape[channel_axis]:
            raise ValueError("channel index is out of range.")
        x = x[channel, :] if channel_axis == 0 else x[:, channel]
    elif channel is not None or channel_axis is not None:
        raise ValueError("channel and channel_axis apply only to 2D input.")
    if not np.issubdtype(x.dtype, np.number) or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("signal must be a non-empty finite numeric array.")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive.")
    if not np.isfinite(threshold_sd) or threshold_sd >= 0:
        raise ValueError("threshold_sd must be finite and negative.")
    x = np.asarray(x, dtype=float)
    noise = estimate_noise_mad(x)
    threshold = threshold_sd * noise
    candidates = np.flatnonzero((x[1:-1] < threshold) &
                                (x[1:-1] <= x[:-2]) &
                                (x[1:-1] < x[2:])) + 1
    times = candidates / float(sampling_rate_hz)
    return {"samples": candidates, "times_s": times,
            "amplitudes": x[candidates], "noise_sigma": noise,
            "threshold": threshold, "threshold_sd": threshold_sd}
