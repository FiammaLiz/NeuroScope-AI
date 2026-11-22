"""Regression tests using HDF5 fixtures that emulate MATLAB v7.3 layouts."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from neuroscope import dataset_loading
from neuroscope.analysis.contracts import (
    MetadataSnapshot,
    OperationContext,
    SampleInterval,
)
from neuroscope.analysis.operations import OperationRunner
from neuroscope.analysis.registry import OperationRegistry
from neuroscope.analysis.signed_peak_operation import OPERATION_ID as SIGNED_ID
from neuroscope.analysis.signed_peak_operation import (
    OPERATION_VERSION as SIGNED_VERSION,
)
from neuroscope.analysis.signed_peak_operation import register_signed_peak_detector
from neuroscope.analysis.spike_operation import OPERATION_ID as NEGATIVE_ID
from neuroscope.analysis.spike_operation import OPERATION_VERSION as NEGATIVE_VERSION
from neuroscope.analysis.spike_operation import register_negative_spike_detector
from neuroscope.data.hdf5_source import (
    inspect_matlab_v73,
    is_matlab_v73,
    open_matlab_v73,
)
from neuroscope.data.loaders import DataLoadError, load_array
from neuroscope.data_access import DatasetAccessError, DatasetStore
from neuroscope.dataset_loading import prepare_dataset
from neuroscope.domain import DatasetMetadata
from neuroscope.exploration import DatasetExploration, MetadataStatus, SampleWindow
from neuroscope.rendering import render_exploration_signal


def _write_matlab_v73_header(path: Path) -> None:
    description = b"MATLAB 7.3 MAT-file, Platform: Synthetic fixture"
    with path.open("r+b") as stream:
        stream.write(description[:116].ljust(116, b" ") + b"\0" * 8 + b"\x00\x02IM")


def _write_emulated_matlab_v73(path: Path, datasets: dict[str, np.ndarray]) -> None:
    """Write a fixture with reversed MATLAB-to-HDF5 dimension ordering.

    This is a synthetic HDF5 fixture, not a MATLAB-generated v7.3 file.
    """
    with h5py.File(path, "w", userblock_size=512) as h5:
        for name, logical in datasets.items():
            data = np.asarray(logical)
            physical = data.transpose(tuple(reversed(range(data.ndim)))) if data.ndim > 1 else data
            item = h5.create_dataset(name, data=physical, chunks=True)
            item.attrs["MATLAB_class"] = np.bytes_("double")
    _write_matlab_v73_header(path)


def test_inspection_is_metadata_only_and_lists_nested_numeric_datasets(tmp_path, monkeypatch):
    path = tmp_path / "recording.mat"
    _write_emulated_matlab_v73(
        path,
        {"recording": np.arange(12).reshape(3, 4), "group/other": np.arange(5)},
    )
    reads = []
    original_getitem = h5py.Dataset.__getitem__

    def track_read(dataset, selection):
        reads.append(selection)
        return original_getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", track_read)

    assert is_matlab_v73(path)
    info = inspect_matlab_v73(path)

    assert reads == []
    assert [(item.path, item.shape) for item in info] == [
        ("group/other", (5,)),
        ("recording", (3, 4)),
    ]


@pytest.mark.parametrize(
    ("name", "logical", "expected_shape"),
    [
        ("row", np.arange(5).reshape(1, 5), (5,)),
        ("column", np.arange(5).reshape(5, 1), (5,)),
        ("matrix", np.arange(12).reshape(3, 4), (3, 4)),
        ("single", np.asarray([[9]]), (1,)),
    ],
)
def test_matlab_logical_shapes_values_and_windows(tmp_path, name, logical, expected_shape):
    path = tmp_path / f"{name}.mat"
    _write_emulated_matlab_v73(path, {"signal": logical})
    source = open_matlab_v73(path, "signal")

    assert source.shape == expected_shape
    if len(expected_shape) == 1:
        loaded = source.read_window(0, expected_shape[0])
        expected = logical.reshape(-1)
    else:
        loaded = source.read_window(0, expected_shape[1], channel_axis=0)
        expected = logical
    np.testing.assert_array_equal(loaded, expected)
    assert loaded.flags.writeable is False
    source.close()


@pytest.mark.parametrize("channel_axis", [0, 1])
def test_selected_channel_reads_use_logical_orientation_and_distant_window(tmp_path, channel_axis):
    path = tmp_path / "channels.mat"
    logical = np.arange(8 * 30_000, dtype=np.float32).reshape(8, 30_000)
    if channel_axis == 1:
        logical = logical.T
    _write_emulated_matlab_v73(path, {"recording": logical})
    source = open_matlab_v73(path, "recording")
    selection_calls = []
    original_getitem = h5py.Dataset.__getitem__

    def track_read(dataset, selection):
        selection_calls.append(selection)
        return original_getitem(dataset, selection)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(h5py.Dataset, "__getitem__", track_read)
        values = source.read_window(
            24_000,
            24_003,
            channel_axis=channel_axis,
            channel_index=5,
        )

    expected = logical[5, 24_000:24_003] if channel_axis == 0 else logical[24_000:24_003, 5]
    np.testing.assert_array_equal(values, expected)
    assert len(values) == 3
    assert len(selection_calls) == 1
    assert all(not isinstance(index, slice) or index.stop - index.start <= 3 for index in selection_calls[0])
    source.close()


def test_unsupported_matlab_character_and_higher_dimensional_datasets_are_not_selected(tmp_path):
    path = tmp_path / "unsupported.mat"
    with h5py.File(path, "w", userblock_size=512) as h5:
        text = h5.create_dataset("text", data=np.arange(4, dtype=np.uint16))
        text.attrs["MATLAB_class"] = np.bytes_("char")
        numeric = h5.create_dataset("volume", data=np.zeros((2, 3, 4)))
        numeric.attrs["MATLAB_class"] = np.bytes_("double")
        struct = h5.create_group("unsupported_struct")
        struct.attrs["MATLAB_class"] = np.bytes_("struct")
        field = struct.create_dataset("field", data=np.arange(3))
        field.attrs["MATLAB_class"] = np.bytes_("double")
    _write_matlab_v73_header(path)

    assert inspect_matlab_v73(path) == ()
    with pytest.raises(DataLoadError, match="No supported numeric"):
        load_array(path)


def test_generic_hdf5_is_not_silently_treated_as_matlab_v73(tmp_path):
    path = tmp_path / "not_matlab.mat"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("signal", data=np.arange(4))

    assert not is_matlab_v73(path)
    with pytest.raises(DataLoadError, match="lacks the MATLAB v7.3"):
        load_array(path)


def test_hdf5_compound_real_imag_dataset_maps_to_numpy_complex(tmp_path):
    path = tmp_path / "complex.mat"
    expected = np.asarray([1 + 2j, 3 - 4j], dtype=np.complex64)
    physical = np.empty(2, dtype=[("real", "<f4"), ("imag", "<f4")])
    physical["real"] = expected.real
    physical["imag"] = expected.imag
    with h5py.File(path, "w", userblock_size=512) as h5:
        item = h5.create_dataset("signal", data=physical)
        item.attrs["MATLAB_class"] = np.bytes_("single")
    _write_matlab_v73_header(path)

    source = open_matlab_v73(path, "signal")

    assert source.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(source.read_window(0, 2), expected)
    source.close()


def test_nonfinite_values_are_rejected_on_requested_window(tmp_path):
    path = tmp_path / "nonfinite.mat"
    _write_emulated_matlab_v73(path, {"signal": np.asarray([0.0, np.inf, 1.0])})
    source = open_matlab_v73(path, "signal")

    with pytest.raises(DataLoadError, match="NaN or infinite"):
        source.read_window(0, 3)
    source.close()


def test_v73_load_array_remains_eager_and_preserves_logical_matrix(tmp_path):
    path = tmp_path / "small.mat"
    values = np.arange(12).reshape(3, 4)
    _write_emulated_matlab_v73(path, {"signal": values})

    loaded = load_array(path)

    assert loaded.shape == values.shape
    np.testing.assert_array_equal(loaded, values)


@pytest.mark.parametrize(
    ("logical", "expected"),
    [
        (np.arange(5).reshape(1, 5), np.arange(5)),
        (np.arange(5).reshape(5, 1), np.arange(5)),
        (np.asarray([[7]]), np.asarray([7])),
    ],
)
def test_v73_eager_loader_preserves_established_mat_vector_shapes(tmp_path, logical, expected):
    path = tmp_path / "vector.mat"
    _write_emulated_matlab_v73(path, {"signal": logical})

    loaded = load_array(path)

    assert loaded.shape == expected.shape
    np.testing.assert_array_equal(loaded, expected)


def test_prepare_dataset_is_lazy_and_preserves_metadata_as_inferred(tmp_path):
    path = tmp_path / "large.mat"
    logical = np.arange(3 * 100_000, dtype=np.float32).reshape(3, 100_000)
    _write_emulated_matlab_v73(path, {"recording": logical})
    with h5py.File(path, "a") as h5:
        h5["recording"].attrs["sampling_rate_hz"] = 20_000.0
        h5["recording"].attrs["units"] = np.bytes_("mV")

    prepared = prepare_dataset(path, confirm_channel_axis=lambda shape: 0)

    assert prepared is not None
    assert prepared.exploration.summary.shape == logical.shape
    assert prepared.profile.file_format == "matlab-v7.3"
    assert prepared.profile.dataset_path == "recording"
    metadata = prepared.exploration.summary.metadata
    assert metadata["sampling_rate_hz"].value == 20_000.0
    assert metadata["sampling_rate_hz"].status is MetadataStatus.INFERRED
    assert metadata["units"].value == "mV"
    assert metadata["units"].status is MetadataStatus.INFERRED
    np.testing.assert_array_equal(
        prepared.exploration.window(
            SampleWindow(99_990, 100_000),
            channel_index=2,
        ),
        logical[2, 99_990:100_000],
    )
    assert prepared.profile.dataset_path == "recording"
    prepared.close()
    assert prepared.array.closed
    with pytest.raises(DataLoadError, match="closed"):
        prepared.exploration.window(
            SampleWindow(0, 1),
            channel_index=0,
        )


def test_cancelled_or_failed_lazy_orientation_confirmation_closes_file(tmp_path, monkeypatch):
    path = tmp_path / "oriented.mat"
    _write_emulated_matlab_v73(path, {"recording": np.arange(12).reshape(3, 4)})
    opened = []
    original_open = dataset_loading.open_matlab_v73

    def track_open(*args):
        source = original_open(*args)
        opened.append(source)
        return source

    monkeypatch.setattr(dataset_loading, "open_matlab_v73", track_open)

    assert prepare_dataset(path, confirm_channel_axis=lambda shape: None) is None
    assert opened[-1].closed
    with pytest.raises(ValueError, match="channel_axis"):
        prepare_dataset(path, confirm_channel_axis=lambda shape: 3)
    assert opened[-1].closed


def test_repeated_reads_reuse_one_open_source(tmp_path):
    path = tmp_path / "repeated.mat"
    values = np.arange(1000)
    _write_emulated_matlab_v73(path, {"signal": values})
    source = open_matlab_v73(path, "signal")
    handle = source._file

    first = source.read_window(100, 120)
    second = source.read_window(900, 920)

    assert source._file is handle
    assert handle.id.valid
    np.testing.assert_array_equal(first, values[100:120])
    np.testing.assert_array_equal(second, values[900:920])
    source.close()


def test_multiple_numeric_datasets_require_explicit_selection_and_cancel_is_safe(tmp_path):
    path = tmp_path / "multiple.mat"
    _write_emulated_matlab_v73(path, {"one": np.arange(4), "group/two": np.arange(7)})
    message = []

    assert prepare_dataset(path, select_mat_variable=lambda text: None) is None
    prepared = prepare_dataset(
        path,
        select_mat_variable=lambda text: message.append(text) or "group/two",
    )

    assert prepared is not None
    assert message and "one" in message[0] and "group/two" in message[0]
    assert prepared.profile.dataset_path == "group/two"
    np.testing.assert_array_equal(prepared.materialize(), np.arange(7))
    prepared.close()


def test_full_recording_preview_uses_bounded_reads_and_preserves_indices(tmp_path, monkeypatch):
    path = tmp_path / "preview.mat"
    values = np.sin(np.arange(30_000, dtype=np.float32) / 10)
    _write_emulated_matlab_v73(path, {"signal": values})
    prepared = prepare_dataset(path)
    assert prepared is not None
    source = prepared.array
    calls = []
    original_read = source.read_window

    def tracked_read(start, stop, **kwargs):
        calls.append((start, stop))
        return original_read(start, stop, **kwargs)

    monkeypatch.setattr(source, "read_window", tracked_read)
    rendered = render_exploration_signal(
        prepared.exploration,
        channel_index=0,
        start_sample=0,
        stop_sample=len(values),
        pixel_width=12,
        chunk_samples=256,
    )

    assert rendered.reduced
    assert len(rendered.values) <= 24
    assert calls and max(stop - start for start, stop in calls) <= 256
    assert rendered.sample_indices.min() >= 0
    assert rendered.sample_indices.max() < len(values)
    prepared.close()


@pytest.mark.parametrize("operation_id,operation_version,register", [
    (NEGATIVE_ID, NEGATIVE_VERSION, register_negative_spike_detector),
    (SIGNED_ID, SIGNED_VERSION, register_signed_peak_detector),
])
def test_both_production_detectors_execute_from_approved_lazy_windows(
    tmp_path, operation_id, operation_version, register
):
    path = tmp_path / "operation.mat"
    values = np.asarray([0.0, -12.0, 0.0, 0.0, -11.0, 0.0])
    _write_emulated_matlab_v73(path, {"recording": values})
    prepared = prepare_dataset(path)
    assert prepared is not None
    profile = prepared.profile.model_copy(
        update={
            "metadata": DatasetMetadata(signal_kind="raw_voltage"),
            "confirmed_fields": ["signal_kind"],
            "unknown_fields": ["sampling_rate_hz", "units", "filtering_description"],
        }
    )
    exploration = DatasetExploration(prepared.array, profile)
    store = DatasetStore()
    registration = store.register(exploration)
    reference = registration.reference((0,), SampleInterval(start=2, stop=6))
    context = OperationContext(
        metadata=tuple(
            MetadataSnapshot(
                name=name,
                value=exploration.summary.metadata[name].value,
                status=exploration.summary.metadata[name].status,
            )
            for name in ("signal_kind", "sampling_rate_hz", "units")
        )
    )
    registry = OperationRegistry()
    register(registry)
    runner = OperationRunner(registry, data_store=store)
    if operation_id == NEGATIVE_ID:
        params = {"threshold_sd": -4.0}
    else:
        params = {"threshold_mode": "absolute", "threshold": -5.0, "noise_override": 1.0}
    request = runner.prepare(
        operation_id,
        operation_version,
        {"signal": reference},
        params,
        context=context,
    )
    approved = runner.approve(request, approved_by="hdf5-test")
    record = runner.execute(approved)

    assert record.status.value == "succeeded"
    assert record.data_references == (reference,)
    assert record.input_data.signal.sample_interval == SampleInterval(start=2, stop=6)
    np.testing.assert_array_equal(
        exploration.window(SampleWindow(start=0, stop=6), channel_index=0), values
    )
    store.unregister(registration.session_handle)
    assert prepared.array.closed


def test_unregister_closes_lazy_source_and_stale_data_reference_fails(tmp_path):
    path = tmp_path / "close.mat"
    _write_emulated_matlab_v73(path, {"signal": np.arange(5)})
    prepared = prepare_dataset(path)
    assert prepared is not None
    store = DatasetStore()
    registration = store.register(prepared.exploration)
    reference = registration.reference((0,), SampleInterval(start=0, stop=2))

    store.unregister(registration.session_handle)

    assert prepared.array.closed
    with pytest.raises(DatasetAccessError, match="unknown or stale"):
        store.resolve(reference)

