import numpy as np
import pytest
from scipy.io import savemat

from neuroscope.dataset_loading import (
    metadata_panel_text,
    plot_axis_labels,
    prepare_dataset,
)
from neuroscope.exploration import DatasetExploration, MetadataStatus


def test_prepare_dataset_requires_orientation_before_creating_profile(tmp_path):
    path = tmp_path / "recording.npy"
    np.save(path, np.arange(12).reshape(3, 4))

    with pytest.raises(ValueError, match="explicit channel orientation"):
        prepare_dataset(path)

    prepared = prepare_dataset(path, confirm_channel_axis=lambda shape: 0)

    assert prepared is not None
    assert prepared.profile.metadata.channel_axis == 0
    assert prepared.exploration.summary.channel_count == 3
    assert prepared.exploration.summary.sample_count == 4


def test_cancelled_orientation_does_not_prepare_a_dataset(tmp_path):
    path = tmp_path / "recording.npy"
    np.save(path, np.zeros((2, 5)))

    assert prepare_dataset(path, confirm_channel_axis=lambda shape: None) is None


def test_mat_variable_selection_is_explicit_and_cancellable(tmp_path):
    path = tmp_path / "recording.mat"
    savemat(path, {"first": np.arange(3), "second": np.arange(4)})
    selected = []

    assert prepare_dataset(path, select_mat_variable=lambda message: None) is None
    prepared = prepare_dataset(
        path,
        select_mat_variable=lambda message: selected.append(message) or "second",
        confirm_channel_axis=lambda shape: 0,
    )

    assert prepared is not None
    assert selected and "multiple numeric variables" in selected[0]
    assert prepared.profile.name == "recording:second"
    np.testing.assert_array_equal(prepared.array, np.arange(4))


def test_metadata_panel_shows_status_and_never_invents_metadata(tmp_path):
    path = tmp_path / "arbitrary.npy"
    np.save(path, np.arange(5, dtype=np.float32))
    prepared = prepare_dataset(path)

    assert prepared is not None
    panel = metadata_panel_text(prepared)
    summary = prepared.exploration.summary

    assert "Original shape: (5,)" in panel
    assert "Data type: float32" in panel
    assert "Channels: 1" in panel
    assert "Samples: 5" in panel
    assert "Sampling rate: UNKNOWN" in panel
    assert "Units: UNKNOWN" in panel
    assert "Signal kind: UNKNOWN" in panel
    assert "Duration" not in panel
    assert "Voltage" not in panel
    assert summary.metadata["sampling_rate_hz"].status is MetadataStatus.UNKNOWN
    assert plot_axis_labels(summary) == ("Sample index", "Value (units unknown)")


def test_confirmed_rate_drives_duration_and_time_axis_label(tmp_path):
    path = tmp_path / "recording.npy"
    values = np.arange(5, dtype=np.float64)
    np.save(path, values)
    prepared = prepare_dataset(path)
    assert prepared is not None
    metadata = prepared.profile.metadata.model_copy(
        update={"sampling_rate_hz": 10.0, "units": "mV"}
    )
    profile = prepared.profile.model_copy(
        update={
            "metadata": metadata,
            "confirmed_fields": ["sampling_rate_hz", "units"],
            "unknown_fields": ["signal_kind"],
        }
    )

    confirmed = type(prepared)(
        prepared.path,
        prepared.array,
        profile,
        DatasetExploration(prepared.array, profile),
    )

    panel = metadata_panel_text(confirmed)
    assert "Sampling rate: 10 [CONFIRMED]" in panel
    assert "Units: mV [CONFIRMED]" in panel
    assert "Nominal duration (N / rate): 0.5 s" in panel
    assert "Final sample timestamp: 0.4 s" in panel
    assert plot_axis_labels(confirmed.exploration.summary) == (
        "Time relative to sample 0 (s)",
        "Value (mV; CONFIRMED)",
    )


def test_inferred_metadata_is_visibly_distinct(tmp_path):
    path = tmp_path / "recording.npy"
    np.save(path, np.arange(3, dtype=np.float64))
    prepared = prepare_dataset(path)
    assert prepared is not None
    profile = prepared.profile.model_copy(
        update={
            "metadata": prepared.profile.metadata.model_copy(update={"units": "mV"}),
            "inferred_fields": ["units"],
            "unknown_fields": ["sampling_rate_hz", "signal_kind"],
        }
    )

    inferred = type(prepared)(
        prepared.path,
        prepared.array,
        profile,
        DatasetExploration(prepared.array, profile),
    )

    assert "Units: mV [INFERRED]" in metadata_panel_text(inferred)

