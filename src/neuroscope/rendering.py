"""Visualization-only sample reduction and original-sample readouts.

The recording remains the scientific source of truth. Display traces carry the
original sample index for every rendered value; reduced samples are never
resampled signal data and must not be passed to analysis operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite

import numpy as np
import numpy.typing as npt

from .exploration import DatasetExploration, MetadataStatus, SampleWindow


@dataclass(frozen=True, slots=True)
class RenderedSignal:
    """Read-only display values and their original sample indices."""

    sample_indices: npt.NDArray[np.int64]
    values: npt.NDArray
    reduced: bool


@dataclass(frozen=True, slots=True)
class SampleReadout:
    """A point read directly from one original channel."""

    channel_index: int
    sample_index: int
    value: int | float
    time_seconds: float | None


def render_signal(
    signal: npt.NDArray,
    *,
    start_sample: int,
    stop_sample: int,
    pixel_width: int,
    direct_samples_per_pixel: float = 2.0,
) -> RenderedSignal:
    """Render a sample interval directly or as a resolution-bounded min/max envelope.

    The visible interval is half-open, ``[start_sample, stop_sample)``. For
    envelope rendering it is divided into fixed-width contiguous buckets of
    ``ceil(visible_count / pixel_width)`` samples. The last bucket may be
    shorter; samples outside the half-open interval are never included and
    no padding is used. Every visible sample belongs to exactly one bucket.
    Each bucket emits its
    earliest minimum and maximum in chronological order; ties choose the
    earliest occurrence, and a shared min/max index is emitted once.

    Windows with at most ``direct_samples_per_pixel * pixel_width`` samples
    retain every original sample. Output arrays are read-only and map each
    display value back to its exact original sample index.
    """
    if not isinstance(signal, np.ndarray) or signal.ndim != 1:
        raise ValueError("signal must be a one-dimensional NumPy array.")
    if not np.issubdtype(signal.dtype, np.number) or np.issubdtype(
        signal.dtype, np.complexfloating
    ):
        raise TypeError("signal must contain real numeric values.")
    for name, value in (("start_sample", start_sample), ("stop_sample", stop_sample)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer sample index.")
    if isinstance(pixel_width, bool) or not isinstance(pixel_width, (int, np.integer)):
        raise TypeError("pixel_width must be a positive integer.")
    if pixel_width <= 0:
        raise ValueError("pixel_width must be positive.")
    if not isfinite(direct_samples_per_pixel) or direct_samples_per_pixel <= 0:
        raise ValueError("direct_samples_per_pixel must be finite and positive.")
    if start_sample < 0 or stop_sample > signal.size or stop_sample <= start_sample:
        raise ValueError("Sample interval must be non-empty and within the signal.")

    start = int(start_sample)
    stop = int(stop_sample)
    width = int(pixel_width)
    visible_count = stop - start
    if not np.isfinite(signal[start:stop]).all():
        raise ValueError("Visible signal samples must be finite.")
    if visible_count <= direct_samples_per_pixel * width:
        indices = np.arange(start, stop, dtype=np.int64)
        values = signal[start:stop].view()
        values.setflags(write=False)
        indices.setflags(write=False)
        return RenderedSignal(indices, values, False)

    output_indices: list[int] = []
    output_values: list[object] = []
    bucket_width = (visible_count + width - 1) // width
    for bucket_start in range(start, stop, bucket_width):
        bucket_stop = min(bucket_start + bucket_width, stop)
        values = signal[bucket_start:bucket_stop]
        min_offset = int(np.argmin(values))
        max_offset = int(np.argmax(values))
        extrema = sorted({min_offset, max_offset})
        for offset in extrema:
            original_index = bucket_start + offset
            output_indices.append(original_index)
            output_values.append(signal[original_index].item())

    indices = np.asarray(output_indices, dtype=np.int64)
    values = np.asarray(output_values, dtype=signal.dtype)
    indices.setflags(write=False)
    values.setflags(write=False)
    return RenderedSignal(indices, values, True)


def render_exploration_signal(
    exploration: DatasetExploration,
    *,
    channel_index: int,
    start_sample: int,
    stop_sample: int,
    pixel_width: int,
    direct_samples_per_pixel: float = 2.0,
    chunk_samples: int = 65_536,
) -> RenderedSignal:
    """Render an exploration window with bounded reads for lazy data sources.

    For a reduced full-recording view, each display bucket is scanned in
    bounded sample chunks and contributes only its earliest minimum and
    maximum. Thus memory is bounded by one input chunk plus the display result.
    """
    if not isinstance(exploration, DatasetExploration):
        raise TypeError("exploration must be a DatasetExploration.")
    if np.issubdtype(np.dtype(exploration.summary.dtype), np.complexfloating):
        raise TypeError("signal must contain real numeric values.")
    if isinstance(pixel_width, bool) or not isinstance(pixel_width, (int, np.integer)):
        raise TypeError("pixel_width must be a positive integer.")
    if pixel_width <= 0:
        raise ValueError("pixel_width must be positive.")
    if not isfinite(direct_samples_per_pixel) or direct_samples_per_pixel <= 0:
        raise ValueError("direct_samples_per_pixel must be finite and positive.")
    if isinstance(chunk_samples, bool) or not isinstance(chunk_samples, (int, np.integer)):
        raise TypeError("chunk_samples must be a positive integer.")
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive.")
    window = SampleWindow(start_sample, stop_sample)
    if window.stop > exploration.summary.sample_count:
        raise ValueError("Sample interval must be within the dataset.")
    visible_count = window.stop - window.start
    if visible_count <= direct_samples_per_pixel * pixel_width:
        values = exploration.window(window, channel_index=channel_index)
        rendered = render_signal(
            values,
            start_sample=0,
            stop_sample=visible_count,
            pixel_width=int(pixel_width),
            direct_samples_per_pixel=direct_samples_per_pixel,
        )
        indices = rendered.sample_indices + window.start
        indices.setflags(write=False)
        return RenderedSignal(indices, rendered.values, rendered.reduced)

    bucket_width = (visible_count + int(pixel_width) - 1) // int(pixel_width)
    output_indices: list[int] = []
    output_values: list[object] = []
    for bucket_start in range(window.start, window.stop, bucket_width):
        bucket_stop = min(bucket_start + bucket_width, window.stop)
        minimum: object | None = None
        maximum: object | None = None
        minimum_index = maximum_index = -1
        for chunk_start, chunk in exploration.iter_channel_chunks(
            SampleWindow(bucket_start, bucket_stop),
            channel_index=channel_index,
            chunk_samples=int(chunk_samples),
        ):
            if not np.isfinite(chunk).all():
                raise ValueError("Visible signal samples must be finite.")
            min_offset = int(np.argmin(chunk))
            max_offset = int(np.argmax(chunk))
            min_value = chunk[min_offset].item()
            max_value = chunk[max_offset].item()
            if minimum is None or min_value < minimum:
                minimum = min_value
                minimum_index = chunk_start + min_offset
            if maximum is None or max_value > maximum:
                maximum = max_value
                maximum_index = chunk_start + max_offset
        extrema = sorted({minimum_index, maximum_index})
        for sample_index in extrema:
            if sample_index == minimum_index:
                value = minimum
            else:
                value = maximum
            output_indices.append(sample_index)
            output_values.append(value)

    indices = np.asarray(output_indices, dtype=np.int64)
    values = np.asarray(output_values, dtype=np.dtype(exploration.summary.dtype))
    indices.setflags(write=False)
    values.setflags(write=False)
    return RenderedSignal(indices, values, True)


def sample_index_from_coordinate(
    coordinate: float,
    *,
    sample_count: int,
    sampling_rate_hz: float | None,
) -> int:
    """Map an axis coordinate to its nearest sample; exact half ties round up.

    Coordinates are seconds when a confirmed rate is supplied and sample
    indices otherwise. Finite coordinates are clamped to the first or final
    sample so axis-edge events cannot produce an out-of-range lookup.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)):
        raise TypeError("sample_count must be a positive integer.")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if not isfinite(coordinate):
        raise ValueError("Cursor coordinate must be finite.")

    coordinate_in_samples = float(coordinate)
    if sampling_rate_hz is not None:
        if not isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be finite and positive.")
        coordinate_in_samples *= sampling_rate_hz
    coordinate_in_samples = min(max(coordinate_in_samples, 0.0), sample_count - 1)
    return floor(coordinate_in_samples + 0.5)


def read_original_sample(
    exploration: DatasetExploration,
    *,
    channel_index: int,
    axis_coordinate: float,
) -> SampleReadout:
    """Read a cursor value directly from the selected original data channel."""
    summary = exploration.summary
    rate_entry = summary.metadata["sampling_rate_hz"]
    confirmed_rate = (
        float(rate_entry.value)
        if rate_entry.status is MetadataStatus.CONFIRMED and rate_entry.value is not None
        else None
    )
    sample_index = sample_index_from_coordinate(
        axis_coordinate,
        sample_count=summary.sample_count,
        sampling_rate_hz=confirmed_rate,
    )
    value = exploration.read_sample(channel_index, sample_index)
    time_seconds = (
        exploration.sample_time_seconds(sample_index)
        if confirmed_rate is not None
        else None
    )
    return SampleReadout(channel_index, sample_index, value, time_seconds)
