from pathlib import Path

import numpy as np
import pytest

from neuroscope.domain import DatasetMetadata, DatasetProfile
from neuroscope.exploration import DatasetExploration, MetadataStatus
from neuroscope.rendering import (
    read_original_sample,
    render_signal,
    sample_index_from_coordinate,
)


def _exploration(
    signal: np.ndarray,
    *,
    channel_axis: int | None = None,
    sampling_rate_hz: float | None = None,
    rate_status: MetadataStatus = MetadataStatus.UNKNOWN,
) -> DatasetExploration:
    values = signal
    profile = DatasetProfile.from_array(
        Path("rendering.npy"), values, channel_axis=channel_axis
    )
    metadata = DatasetMetadata(
        sampling_rate_hz=sampling_rate_hz,
        channel_axis=channel_axis,
        time_axis=1 - channel_axis if channel_axis is not None else None,
    )
    profile = profile.model_copy(
        update={
            "metadata": metadata,
            "confirmed_fields": (["channel_axis"] if channel_axis is not None else [])
            + (["sampling_rate_hz"] if rate_status is MetadataStatus.CONFIRMED else []),
            "inferred_fields": (
                ["sampling_rate_hz"] if rate_status is MetadataStatus.INFERRED else []
            ),
            "unknown_fields": (
                ["sampling_rate_hz"] if rate_status is MetadataStatus.UNKNOWN else []
            )
            + ["units", "signal_kind"],
        }
    )
    return DatasetExploration(values, profile)


def _render(signal, *, width, start=0, stop=None):
    if stop is None:
        stop = len(signal)
    return render_signal(
        signal,
        start_sample=start,
        stop_sample=stop,
        pixel_width=width,
    )


def test_small_window_keeps_every_original_sample_as_a_read_only_view():
    original = np.arange(15, dtype=np.float32)

    rendered = _render(original, width=10)

    assert not rendered.reduced
    np.testing.assert_array_equal(rendered.sample_indices, np.arange(15))
    np.testing.assert_array_equal(rendered.values, original)
    assert not rendered.values.flags.writeable
    assert np.shares_memory(rendered.values, original)


def test_constant_signal_emits_deterministic_single_extremum_per_bucket():
    rendered = _render(np.full(100, 4.0), width=4)

    assert rendered.reduced
    np.testing.assert_array_equal(rendered.sample_indices, [0, 25, 50, 75])
    np.testing.assert_array_equal(rendered.values, [4.0, 4.0, 4.0, 4.0])


@pytest.mark.parametrize(
    "signal, expected_values",
    [
        (np.arange(20), [0, 9, 10, 19]),
        (np.arange(20)[::-1], [19, 10, 9, 0]),
    ],
)
def test_increasing_and_decreasing_buckets_emit_extrema_chronologically(
    signal, expected_values
):
    rendered = _render(signal, width=2)

    np.testing.assert_array_equal(rendered.sample_indices, [0, 9, 10, 19])
    np.testing.assert_array_equal(rendered.values, expected_values)


def test_narrow_positive_and_negative_spikes_survive_minmax_reduction():
    signal = np.zeros(1000, dtype=np.float64)
    signal[202] = 9.0
    signal[205] = -8.0
    signal[808] = 11.0

    rendered = _render(signal, width=5)

    points = dict(zip(rendered.sample_indices.tolist(), rendered.values.tolist()))
    assert points[202] == 9.0
    assert points[205] == -8.0
    assert points[808] == 11.0
    assert np.all(np.diff(rendered.sample_indices) > 0)


def test_ties_choose_earliest_extrema_and_emit_them_once_when_shared():
    signal = np.ones(40, dtype=np.float64)
    signal[3] = -2
    signal[8] = -2
    signal[12] = 5
    signal[18] = 5
    rendered = _render(signal, width=1)

    np.testing.assert_array_equal(rendered.sample_indices, [3, 12])
    np.testing.assert_array_equal(rendered.values, [-2, 5])

    constant = _render(np.zeros(40), width=1)
    np.testing.assert_array_equal(constant.sample_indices, [0])


def test_partial_final_bucket_and_nonzero_interval_boundaries():
    signal = np.arange(20, dtype=np.int64)

    rendered = _render(signal, width=3, start=2, stop=13)

    # Buckets are [2, 6), [6, 10), and the partial final bucket [10, 13).
    np.testing.assert_array_equal(rendered.sample_indices, [2, 5, 6, 9, 10, 12])
    assert rendered.sample_indices.min() >= 2
    assert rendered.sample_indices.max() < 13


def test_point_count_tracks_resolution_and_reduction_mapping_stays_exact():
    signal = np.sin(np.arange(5000) / 17)
    narrow = _render(signal, width=10)
    wide = _render(signal, width=80)

    assert len(narrow.values) <= 2 * 10
    assert len(wide.values) <= 2 * 80
    assert len(wide.values) > len(narrow.values)
    assert np.all((wide.sample_indices >= 0) & (wide.sample_indices < len(signal)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_sample": -1, "stop_sample": 2, "pixel_width": 1},
        {"start_sample": 2, "stop_sample": 2, "pixel_width": 1},
        {"start_sample": 0, "stop_sample": 11, "pixel_width": 1},
        {"start_sample": 0, "stop_sample": 10, "pixel_width": 0},
    ],
)
def test_rendering_rejects_invalid_intervals_or_resolution(kwargs):
    with pytest.raises((TypeError, ValueError)):
        render_signal(np.arange(10), **kwargs)


def test_only_visible_values_are_reduced_and_must_be_finite():
    signal = np.arange(10, dtype=np.float64)
    signal[9] = np.nan

    visible = _render(signal, width=10, start=0, stop=5)
    assert len(visible.values) == 5
    with pytest.raises(ValueError, match="Visible signal samples must be finite"):
        _render(signal, width=10, start=5, stop=10)


def test_original_values_and_recording_remain_unchanged_after_reduction():
    signal = np.arange(2000, dtype=np.float64)
    before = signal.copy()
    rendered = _render(signal, width=10)
    exploration = _exploration(signal)
    original_index = int(rendered.sample_indices[len(rendered.sample_indices) // 2])
    readout = read_original_sample(
        exploration,
        channel_index=0,
        axis_coordinate=float(original_index),
    )

    assert readout.sample_index == original_index
    assert readout.value == signal[readout.sample_index]
    np.testing.assert_array_equal(signal, before)


def test_cursor_mapping_uses_nearest_sample_half_up_and_clamps_boundaries():
    assert sample_index_from_coordinate(
        2.49, sample_count=10, sampling_rate_hz=None
    ) == 2
    assert sample_index_from_coordinate(
        2.5, sample_count=10, sampling_rate_hz=None
    ) == 3
    assert sample_index_from_coordinate(
        0.25, sample_count=10, sampling_rate_hz=10.0
    ) == 3
    assert sample_index_from_coordinate(
        -1.0, sample_count=10, sampling_rate_hz=None
    ) == 0
    assert sample_index_from_coordinate(
        100.0, sample_count=10, sampling_rate_hz=None
    ) == 9


def test_readout_includes_seconds_only_for_confirmed_rate():
    values = np.arange(10, dtype=np.float64)
    unknown = _exploration(values)
    unknown_readout = read_original_sample(
        unknown, channel_index=0, axis_coordinate=3.2
    )
    assert unknown_readout.sample_index == 3
    assert unknown_readout.value == 3.0
    assert unknown_readout.time_seconds is None

    confirmed = _exploration(
        values,
        sampling_rate_hz=10.0,
        rate_status=MetadataStatus.CONFIRMED,
    )
    confirmed_readout = read_original_sample(
        confirmed, channel_index=0, axis_coordinate=0.35
    )
    assert confirmed_readout.sample_index == 4
    assert confirmed_readout.value == 4.0
    assert confirmed_readout.time_seconds == 0.4


def test_readout_keeps_original_channel_index_for_multichannel_data():
    values = np.arange(30, dtype=np.int64).reshape(3, 10)
    exploration = _exploration(values, channel_axis=0)

    readout = read_original_sample(
        exploration, channel_index=2, axis_coordinate=6.0
    )

    assert (readout.channel_index, readout.sample_index, readout.value) == (2, 6, 26)

