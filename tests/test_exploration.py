from math import inf
from pathlib import Path

import numpy as np
import pytest

from neuroscope.domain import DatasetMetadata, DatasetProfile
from neuroscope.exploration import (
    DatasetExploration,
    MetadataStatus,
    SampleWindow,
)


def _profile(
    array: np.ndarray,
    *,
    channel_axis: int | None = None,
    sampling_rate_hz: float | None = None,
    confirmed_fields: tuple[str, ...] = (),
    inferred_fields: tuple[str, ...] = (),
    unknown_fields: tuple[str, ...] = (),
    units: str | None = None,
    signal_kind: str | None = None,
) -> DatasetProfile:
    profile = DatasetProfile.from_array(
        Path("recording.npy"), array, channel_axis=channel_axis
    )
    metadata = DatasetMetadata(
        sampling_rate_hz=sampling_rate_hz,
        units=units,
        signal_kind=signal_kind,
        channel_axis=channel_axis,
        time_axis=1 - channel_axis if channel_axis is not None else None,
    )
    return profile.model_copy(
        update={
            "metadata": metadata,
            "confirmed_fields": list(confirmed_fields),
            "inferred_fields": list(inferred_fields),
            "unknown_fields": list(unknown_fields),
        }
    )


def _exploration(
    array: np.ndarray,
    *,
    channel_axis: int | None = None,
    sampling_rate_hz: float | None = None,
    rate_status: MetadataStatus = MetadataStatus.UNKNOWN,
    units: str | None = None,
    unit_status: MetadataStatus = MetadataStatus.UNKNOWN,
    signal_kind: str | None = None,
    signal_kind_status: MetadataStatus = MetadataStatus.UNKNOWN,
) -> DatasetExploration:
    confirmed = []
    inferred = []
    unknown = ["acquisition_origin"]
    if array.ndim == 2:
        confirmed.append("channel_axis")
    for field, status in (
        ("sampling_rate_hz", rate_status),
        ("units", unit_status),
        ("signal_kind", signal_kind_status),
    ):
        if status is MetadataStatus.CONFIRMED:
            confirmed.append(field)
        elif status is MetadataStatus.INFERRED:
            inferred.append(field)
        else:
            unknown.append(field)
    profile = _profile(
        array,
        channel_axis=channel_axis,
        sampling_rate_hz=sampling_rate_hz,
        confirmed_fields=tuple(confirmed),
        inferred_fields=tuple(inferred),
        unknown_fields=tuple(unknown),
        units=units,
        signal_kind=signal_kind,
    )
    return DatasetExploration(array, profile)


def test_one_dimensional_summary_and_unknown_metadata():
    values = np.arange(5, dtype=np.int16)
    exploration = _exploration(values)
    summary = exploration.summary

    assert summary.shape == (5,)
    assert summary.dtype == "int16"
    assert summary.channel_count == 1
    assert summary.sample_count == 5
    assert summary.channel_axis is None
    assert summary.sample_axis == 0
    assert summary.metadata["sampling_rate_hz"].status is MetadataStatus.UNKNOWN
    assert summary.metadata["units"].status is MetadataStatus.UNKNOWN
    assert summary.metadata["signal_kind"].status is MetadataStatus.UNKNOWN
    assert summary.metadata["acquisition_origin"].value is None
    assert summary.metadata["acquisition_origin"].status is MetadataStatus.UNKNOWN
    assert summary.nominal_duration_s is None
    assert summary.final_sample_time_s is None
    np.testing.assert_array_equal(exploration.channel(0), values)
    with pytest.raises(ValueError, match="sampling_rate_hz is confirmed"):
        exploration.sample_time_seconds(0)


@pytest.mark.parametrize(
    "channel_axis, expected_channels, expected_samples, expected_channel",
    [
        (0, 3, 4, [4, 5, 6, 7]),
        (1, 4, 3, [1, 5, 9]),
    ],
)
def test_two_dimensional_orientation_and_channel_views(
    channel_axis, expected_channels, expected_samples, expected_channel
):
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    exploration = _exploration(values, channel_axis=channel_axis)
    summary = exploration.summary

    assert summary.shape == (3, 4)
    assert summary.channel_count == expected_channels
    assert summary.sample_count == expected_samples
    assert summary.channel_axis == channel_axis
    assert summary.sample_axis == 1 - channel_axis
    assert summary.metadata["channel_axis"].status is MetadataStatus.CONFIRMED
    assert summary.metadata["time_axis"].value == 1 - channel_axis
    assert summary.metadata["time_axis"].status is MetadataStatus.INFERRED
    np.testing.assert_array_equal(exploration.channel(1), expected_channel)


def test_confirmed_inferred_and_unknown_metadata_are_distinct():
    values = np.zeros((2, 6), dtype=np.float32)
    exploration = _exploration(
        values,
        channel_axis=0,
        sampling_rate_hz=100.0,
        rate_status=MetadataStatus.CONFIRMED,
        units="mV",
        unit_status=MetadataStatus.INFERRED,
        signal_kind=None,
        signal_kind_status=MetadataStatus.UNKNOWN,
    )
    metadata = exploration.summary.metadata

    assert metadata["sampling_rate_hz"].value == 100.0
    assert metadata["sampling_rate_hz"].status is MetadataStatus.CONFIRMED
    assert metadata["units"].value == "mV"
    assert metadata["units"].status is MetadataStatus.INFERRED
    assert metadata["signal_kind"].value is None
    assert metadata["signal_kind"].status is MetadataStatus.UNKNOWN


def test_duration_and_last_sample_time_use_confirmed_rate_only():
    values = np.arange(5, dtype=np.float64)
    at_10_hz = _exploration(
        values, sampling_rate_hz=10.0, rate_status=MetadataStatus.CONFIRMED
    )
    at_20_hz = _exploration(
        values, sampling_rate_hz=20.0, rate_status=MetadataStatus.CONFIRMED
    )
    inferred_rate = _exploration(
        values, sampling_rate_hz=10.0, rate_status=MetadataStatus.INFERRED
    )

    assert at_10_hz.summary.nominal_duration_s == 0.5
    assert at_10_hz.summary.final_sample_time_s == 0.4
    assert at_10_hz.summary.metadata["nominal_duration_s"].status is MetadataStatus.INFERRED
    assert at_10_hz.summary.metadata["final_sample_time_s"].status is MetadataStatus.INFERRED
    assert at_10_hz.sample_time_seconds(2) == 0.2
    assert at_20_hz.summary.nominal_duration_s == 0.25
    assert at_20_hz.summary.final_sample_time_s == 0.2
    assert at_20_hz.sample_time_seconds(2) == 0.1
    assert inferred_rate.summary.nominal_duration_s is None
    assert inferred_rate.summary.final_sample_time_s is None
    with pytest.raises(ValueError, match="sampling_rate_hz is confirmed"):
        inferred_rate.sample_time_seconds(2)

    with pytest.raises(ValueError, match="finite and positive"):
        _exploration(
            values, sampling_rate_hz=inf, rate_status=MetadataStatus.CONFIRMED
        )


def test_sample_times_are_relative_and_do_not_add_unknown_origin():
    values = np.arange(4, dtype=np.float64)
    exploration = _exploration(
        values, sampling_rate_hz=4.0, rate_status=MetadataStatus.CONFIRMED
    )

    assert exploration.summary.metadata["acquisition_origin"].value is None
    assert exploration.sample_time_seconds(0) == 0.0
    assert exploration.sample_time_seconds(3) == 0.75


def test_channel_indices_are_validated():
    exploration = _exploration(np.arange(5))

    with pytest.raises(IndexError, match="outside"):
        exploration.channel(-1)
    with pytest.raises(IndexError, match="outside"):
        exploration.channel(1)
    with pytest.raises(TypeError, match="integer"):
        exploration.channel(0.0)
    with pytest.raises(TypeError, match="integer"):
        exploration.channel(True)


def test_two_dimensional_arrays_require_confirmed_orientation():
    values = np.zeros((2, 3))

    with pytest.raises(ValueError, match="Confirm channel_axis"):
        DatasetExploration.from_array(values)
    with pytest.raises(ValueError, match="Confirm channel_axis"):
        DatasetExploration.from_array(values, channel_axis=True)
    with pytest.raises(ValueError, match="explicitly confirmed"):
        DatasetExploration(values, _profile(values, channel_axis=0))


def test_windows_are_half_open_bounded_and_preserve_original_indices():
    values = np.arange(12, dtype=np.int64).reshape(3, 4)
    exploration = _exploration(
        values,
        channel_axis=0,
        sampling_rate_hz=10.0,
        rate_status=MetadataStatus.CONFIRMED,
    )
    window = SampleWindow(1, 4)
    selected = exploration.window(window, channel_index=1)

    np.testing.assert_array_equal(selected, [5, 6, 7])
    assert exploration.sample_time_seconds(window.start) == 0.1
    full = exploration.window(exploration.full_window)
    np.testing.assert_array_equal(full, values)
    assert full.shape == values.shape
    assert np.shares_memory(selected, values)


def test_invalid_or_empty_windows_are_rejected():
    values = np.arange(5)
    exploration = _exploration(values)

    with pytest.raises(ValueError, match="non-negative"):
        SampleWindow(-1, 2)
    with pytest.raises(ValueError, match="non-empty"):
        SampleWindow(2, 2)
    with pytest.raises(ValueError, match="non-empty"):
        SampleWindow(3, 2)
    with pytest.raises(ValueError, match="exceeds"):
        exploration.window(SampleWindow(2, 6))


def test_selected_views_are_read_only_and_source_values_remain_unchanged():
    values = np.arange(12, dtype=np.int16).reshape(3, 4)
    original = values.copy()
    exploration = _exploration(values, channel_axis=0)
    view = exploration.window(SampleWindow(1, 3), channel_index=1)

    assert np.shares_memory(view, values)
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view[0] = -100
    np.testing.assert_array_equal(values, original)
    assert values.flags.writeable


@pytest.mark.parametrize("channel_axis", [0, 1])
def test_two_dimensional_sample_windows_keep_original_axis_order(channel_axis):
    values = np.arange(12).reshape(3, 4)
    exploration = _exploration(values, channel_axis=channel_axis)
    view = exploration.window(SampleWindow(1, 3))

    expected = values[:, 1:3] if channel_axis == 0 else values[1:3, :]
    np.testing.assert_array_equal(view, expected)
    assert np.shares_memory(view, values)
