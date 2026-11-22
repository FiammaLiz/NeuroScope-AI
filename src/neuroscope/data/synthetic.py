"""Deterministic synthetic continuous voltage recordings for validation.

The recording, generation configuration, and ground-truth events are returned
as separate values. Event timestamps refer to the sample carrying the template
peak, quantized to the nearest sample at the configured sampling rate.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Literal

import numpy as np

from ..domain import DatasetMetadata

Polarity = Literal["positive", "negative"]


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    return converted


@dataclass(frozen=True, slots=True)
class SpikeTemplate:
    """Dimensionless waveform shape with a unit positive peak."""

    template_id: str
    samples: tuple[float, ...]
    peak_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.template_id, str) or not self.template_id:
            raise ValueError("template_id must be a non-empty string.")
        try:
            values = tuple(float(value) for value in self.samples)
        except (TypeError, ValueError) as exc:
            raise ValueError("template samples must be a non-empty numeric sequence.") from exc
        if not values:
            raise ValueError("template samples must be non-empty.")
        if not np.isfinite(values).all():
            raise ValueError("template samples must be finite.")
        object.__setattr__(self, "samples", values)
        if not isinstance(self.peak_index, int) or isinstance(self.peak_index, bool):
            raise TypeError("peak_index must be an integer sample index.")
        if not 0 <= self.peak_index < len(self.samples):
            raise ValueError("peak_index is outside the template.")
        if (
            not np.isclose(values[self.peak_index], 1.0)
            or not np.isclose(np.max(np.abs(values)), 1.0)
            or int(np.argmax(np.abs(values))) != self.peak_index
        ):
            raise ValueError("template peak sample must be its unit positive maximum (1.0).")


@dataclass(frozen=True, slots=True)
class SyntheticSpike:
    """Requested synthetic event; timestamp denotes the template peak in seconds."""

    spike_id: str
    channel_index: int
    timestamp_s: float
    amplitude: float
    template_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.spike_id, str) or not self.spike_id:
            raise ValueError("spike_id must be a non-empty string.")
        if not isinstance(self.channel_index, int) or isinstance(self.channel_index, bool):
            raise TypeError("channel_index must be an integer (zero-based).")
        object.__setattr__(self, "timestamp_s", _finite_number(self.timestamp_s, "spike timestamp_s"))
        object.__setattr__(self, "amplitude", _finite_number(self.amplitude, "spike amplitude"))
        if self.amplitude == 0:
            raise ValueError("spike amplitude must be finite and non-zero.")
        if not isinstance(self.template_id, str) or not self.template_id:
            raise ValueError("template_id must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class SyntheticRecordingConfig:
    """Complete, explicit configuration for one generated recording."""

    channel_count: int
    duration_s: float
    sampling_rate_hz: float
    voltage_units: str
    seed: int
    noise_std_by_channel: tuple[float, ...]
    templates: tuple[SpikeTemplate, ...]
    spikes: tuple[SyntheticSpike, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.channel_count, int) or isinstance(self.channel_count, bool) or self.channel_count < 1:
            raise ValueError("channel_count must be a positive integer.")
        duration_s = _finite_number(self.duration_s, "duration_s")
        sampling_rate_hz = _finite_number(self.sampling_rate_hz, "sampling_rate_hz")
        if duration_s <= 0:
            raise ValueError("duration_s must be finite and positive.")
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be finite and positive.")
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "sampling_rate_hz", sampling_rate_hz)
        if not isinstance(self.voltage_units, str) or not self.voltage_units.strip():
            raise ValueError("voltage_units must be a non-empty string.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        noise_values = tuple(_finite_number(value, "noise standard deviation") for value in self.noise_std_by_channel)
        if len(noise_values) != self.channel_count:
            raise ValueError("noise_std_by_channel must contain one value per channel.")
        noise = np.asarray(noise_values, dtype=float)
        if not np.isfinite(noise).all() or np.any(noise < 0):
            raise ValueError("per-channel noise standard deviations must be finite and non-negative.")
        object.__setattr__(self, "noise_std_by_channel", noise_values)
        templates = tuple(self.templates)
        spikes = tuple(self.spikes)
        object.__setattr__(self, "templates", templates)
        object.__setattr__(self, "spikes", spikes)
        if any(not isinstance(template, SpikeTemplate) for template in templates):
            raise TypeError("templates must contain SpikeTemplate values.")
        if any(not isinstance(spike, SyntheticSpike) for spike in spikes):
            raise TypeError("spikes must contain SyntheticSpike values.")
        template_ids = [template.template_id for template in templates]
        if not template_ids or len(set(template_ids)) != len(template_ids):
            raise ValueError("templates must be non-empty and have unique template_id values.")
        spike_ids = [spike.spike_id for spike in spikes]
        if len(set(spike_ids)) != len(spike_ids):
            raise ValueError("spike_id values must be unique.")
        template_by_id = {template.template_id: template for template in self.templates}
        sample_count = self.sample_count
        for spike in self.spikes:
            if not 0 <= spike.channel_index < self.channel_count:
                raise ValueError(f"spike {spike.spike_id!r} channel_index is out of range.")
            if not 0 <= spike.timestamp_s < self.duration_s:
                raise ValueError(f"spike {spike.spike_id!r} timestamp is outside the recording.")
            if spike.template_id not in template_by_id:
                raise ValueError(f"spike {spike.spike_id!r} references an unknown template.")
            peak_sample = self.timestamp_to_sample(spike.timestamp_s)
            template = template_by_id[spike.template_id]
            start = peak_sample - template.peak_index
            stop = start + len(template.samples)
            if start < 0 or stop > sample_count:
                raise ValueError(f"spike {spike.spike_id!r} waveform extends beyond recording boundaries.")

    @property
    def sample_count(self) -> int:
        count = self.duration_s * self.sampling_rate_hz
        rounded = round(count)
        if not np.isclose(count, rounded, rtol=0.0, atol=1e-9):
            raise ValueError("duration_s * sampling_rate_hz must be an integer sample count.")
        return rounded

    def timestamp_to_sample(self, timestamp_s: float) -> int:
        """Map non-negative time to the nearest sample (ties round upward)."""
        return int(np.floor(timestamp_s * self.sampling_rate_hz + 0.5))


@dataclass(frozen=True, slots=True)
class GroundTruthSpike:
    spike_id: str
    channel_index: int
    sample_index: int
    timestamp_s: float
    amplitude: float
    polarity: Polarity
    template_id: str


@dataclass(frozen=True, slots=True)
class SyntheticRecording:
    """Generated samples and the explicit configuration that produced them."""

    signal: np.ndarray
    metadata: DatasetMetadata
    config: SyntheticRecordingConfig

    def save_npy(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save samples to ``.npy``; replace an existing file only when requested.

        With the default ``overwrite=False``, creation is exclusive and an existing
        destination is left untouched. ``overwrite=True`` explicitly replaces the
        destination atomically, so callers must choose a path they intend to
        replace. A non-``.npy`` destination is always rejected.
        """
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a bool.")
        path = Path(path)
        if path.suffix.lower() != ".npy":
            raise ValueError("Synthetic recordings must be saved with a .npy extension.")
        if not overwrite:
            with path.open("xb") as stream:
                np.save(stream, self.signal, allow_pickle=False)
            return path

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                np.save(stream, self.signal, allow_pickle=False)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return path


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    recording: SyntheticRecording
    ground_truth: tuple[GroundTruthSpike, ...]


def generate_synthetic_recording(config: SyntheticRecordingConfig) -> SyntheticDataset:
    """Generate channel-by-sample voltage and separate, known spike truth.

    Spike amplitude is signed in ``voltage_units``. The peak sample index is
    the timestamp reference. ``timestamp_s`` in ground truth is exactly
    ``sample_index / sampling_rate_hz`` after nearest-sample quantization.
    Background noise is zero-mean Gaussian with independently configured
    standard deviation for each channel.
    """
    rng = np.random.default_rng(config.seed)
    shape = (config.channel_count, config.sample_count)
    signal = np.empty(shape, dtype=np.float64)
    for channel, noise_std in enumerate(config.noise_std_by_channel):
        signal[channel] = rng.normal(0.0, noise_std, config.sample_count)

    templates = {template.template_id: template for template in config.templates}
    truth: list[GroundTruthSpike] = []
    for spike in config.spikes:
        template = templates[spike.template_id]
        peak = config.timestamp_to_sample(spike.timestamp_s)
        start = peak - template.peak_index
        stop = start + len(template.samples)
        signal[spike.channel_index, start:stop] += (
            np.asarray(template.samples, dtype=float) * spike.amplitude
        )
        truth.append(
            GroundTruthSpike(
                spike_id=spike.spike_id,
                channel_index=spike.channel_index,
                sample_index=peak,
                timestamp_s=peak / config.sampling_rate_hz,
                amplitude=spike.amplitude,
                polarity="positive" if spike.amplitude > 0 else "negative",
                template_id=spike.template_id,
            )
        )

    signal.setflags(write=False)
    recording = SyntheticRecording(
        signal=signal,
        metadata=DatasetMetadata(
            sampling_rate_hz=config.sampling_rate_hz,
            units=config.voltage_units,
            signal_kind="raw_voltage",
            channel_axis=0,
            time_axis=1,
        ),
        config=config,
    )
    return SyntheticDataset(recording=recording, ground_truth=tuple(truth))
