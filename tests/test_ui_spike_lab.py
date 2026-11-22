"""Headless integration tests for the approval-bound interactive Spike Lab."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from neuroscope.analysis.contracts import SampleInterval
from neuroscope.analysis.operations import OperationRunner
from neuroscope.analysis.signed_peak_operation import OPERATION_ID as SIGNED_ID
from neuroscope.analysis.spike_operation import OPERATION_ID as NEGATIVE_ID
from neuroscope.data.hdf5_source import open_matlab_v73
from neuroscope.data_access import DatasetAccessError
from neuroscope.dataset_loading import PreparedDataset, prepare_dataset
from neuroscope.domain import DatasetProfile
from neuroscope.exploration import DatasetExploration
from neuroscope.memory.project_store import ProjectStore
from neuroscope.ui import main_window as main_window_module
from neuroscope.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.processEvents()


@pytest.fixture
def window(app):
    instance = MainWindow()
    yield instance
    if instance._execution_in_flight:
        _wait_for_worker(instance, app)
    instance.close()
    app.processEvents()


def _dataset(
    values: np.ndarray,
    path: Path,
    *,
    sampling_rate: float | None = None,
    filtering_description: str | None = None,
    channel_axis: int | None = None,
    units: str | None = None,
):
    values = np.asarray(values)
    profile = DatasetProfile.from_array(path, values, channel_axis=channel_axis)
    metadata = profile.metadata.model_copy(
        update={
            "signal_kind": "raw_voltage",
            "sampling_rate_hz": sampling_rate,
            "units": units,
            "filtering_description": filtering_description,
        }
    )
    confirmed = ["signal_kind"]
    unknown = ["units", "filtering_description", "sampling_rate_hz"]
    if units is not None:
        confirmed.append("units")
        unknown.remove("units")
    if channel_axis is not None:
        confirmed.append("channel_axis")
    if sampling_rate is not None:
        confirmed.append("sampling_rate_hz")
        unknown.remove("sampling_rate_hz")
    if filtering_description is not None:
        confirmed.append("filtering_description")
        unknown.remove("filtering_description")
    profile = profile.model_copy(
        update={
            "metadata": metadata,
            "confirmed_fields": confirmed,
            "inferred_fields": [],
            "unknown_fields": unknown,
        }
    )
    exploration = DatasetExploration(values, profile)
    return PreparedDataset(path, values, profile, exploration)


def _install_dataset(
    window: MainWindow,
    values: np.ndarray,
    tmp_path: Path,
    *,
    fs=None,
    channel_axis=None,
    units=None,
    filtering_description=None,
):
    dataset = _dataset(
        values,
        tmp_path / "synthetic.npy",
        sampling_rate=fs,
        channel_axis=channel_axis,
        units=units,
        filtering_description=filtering_description,
    )
    window._commit_dataset(dataset)
    return dataset


def _install_configured_dataset(
    window: MainWindow,
    values: np.ndarray,
    tmp_path: Path,
    *,
    fs: float | None = None,
    filtering_description: str | None = None,
):
    dataset = _dataset(
        values,
        tmp_path / "synthetic.npy",
        sampling_rate=fs,
        filtering_description=filtering_description,
    )
    window._commit_dataset(dataset)
    return dataset


def _select_project(window: MainWindow, path: Path):
    window.store = ProjectStore(path)
    window.project_path_label.setText(str(path))
    window.operation_runner = OperationRunner(
        window.operation_registry,
        data_store=window.dataset_store,
        artifact_store=window.store.artifacts,
        isi_artifact_store=window.store.isi_artifacts,
        mua_artifact_store=window.store.mua_artifacts,
    )
    window._update_action_state()


def _approve_dialogs(monkeypatch):
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("Test Researcher", True),
    )


def _wait_for_worker(window: MainWindow, app: QApplication, timeout: float = 8.0):
    deadline = time.monotonic() + timeout
    while window._execution_in_flight and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert not window._execution_in_flight, "approved operation worker did not finish"


def _wait_for_view(window: MainWindow, app: QApplication, name: str, timeout: float = 8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        result = {
            "waveform": window._waveform_artifact_view,
            "isi": window._isi_artifact_view,
            "mua": window._mua_artifact_view or window._mua_selection_views,
        }[name]
        running = any(thread.isRunning() for thread, _worker in window._view_workers)
        if result is not None and not running:
            return
        time.sleep(0.005)
    app.processEvents()
    label = {
        "waveform": window.waveform_display_label.text(),
        "isi": window.isi_display_label.text(),
        "mua": window.mua_display_label.text(),
    }[name]
    assert {
        "waveform": window._waveform_artifact_view,
        "isi": window._isi_artifact_view,
        "mua": window._mua_artifact_view or window._mua_selection_views,
    }[name] is not None, f"{name} artifact worker did not finish: {label}"


def _zero_noise_signed_signal() -> np.ndarray:
    values = np.tile(np.asarray([-0.1, 0.1]), 50).astype(np.float64)
    values[25] = 2.0
    values[70] = -2.0
    return values


def test_registry_and_detector_specific_controls(window, tmp_path):
    _install_dataset(window, _zero_noise_signed_signal(), tmp_path)

    assert window.operation_registry.get(NEGATIVE_ID).operation_id == NEGATIVE_ID
    assert window.operation_registry.get(SIGNED_ID).operation_id == SIGNED_ID
    assert not window.negative_controls.isHidden()
    assert window.detector_box.currentData() == NEGATIVE_ID

    window.detector_box.setCurrentIndex(1)
    assert not window.signed_threshold.isHidden()
    window.signed_mode_box.setCurrentIndex(1)
    window._update_detector_controls()
    assert not window.noise_override.isEnabled()
    assert "uncentered" in window.detector_explanation.text().lower()


def test_prepare_is_side_effect_free_and_binds_visible_interval(window, tmp_path, monkeypatch):
    values = np.random.default_rng(17).normal(0, 0.1, 100)
    values[30] = -8
    _install_dataset(window, values, tmp_path, fs=1000)
    window.navigation_state.zoom(20, 60)
    window._redraw_navigation_view()
    calls = []
    monkeypatch.setattr(window.operation_runner, "execute", lambda *args: calls.append(args))

    window.prepare_spike_detection()

    assert calls == []
    assert window._prepared_execution is not None
    prepared = window._prepared_execution
    assert prepared.input_data.signal.channels == (0,)
    assert prepared.input_data.signal.sample_interval.start == 20
    assert prepared.input_data.signal.sample_interval.stop == 60
    review = window.operation_review.toPlainText().lower()
    metadata = {item.name: item for item in prepared.context.metadata}
    assert metadata["sampling_rate_hz"].value == 1000.0
    assert metadata["sampling_rate_hz"].status.value == "confirmed"
    assert metadata["units"].value is None
    assert metadata["units"].status.value == "unknown"
    assert "sampling_rate_hz" in review and "metadata snapshot" in review
    assert window.approve_button.isEnabled() is False  # no project selected


def test_parameter_change_invalidates_prepared_approval(window, tmp_path):
    values = np.random.default_rng(19).normal(0, 0.1, 100)
    _install_dataset(window, values, tmp_path)
    window.prepare_spike_detection()
    assert window._prepared_execution is not None

    window.threshold.setText("-5")

    assert window._prepared_execution is None
    assert "invalidated" in window.operation_review.toPlainText().lower()


def test_invalid_gui_parameter_is_rejected_before_preparation(window, tmp_path):
    _install_dataset(window, np.zeros(40), tmp_path)
    window.threshold.setText("not-a-number")

    window.prepare_spike_detection()

    assert window._prepared_execution is None
    assert "preparation failed" in window.operation_review.toPlainText().lower()
    assert "threshold_sd must be a numeric value" in window.operation_review.toPlainText().lower()


def test_approval_requires_a_project_and_does_not_execute(window, tmp_path, monkeypatch):
    _install_dataset(window, np.zeros(40), tmp_path)
    calls = []
    monkeypatch.setattr(window.operation_runner, "execute", lambda *args: calls.append(args))
    messages = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda *args: messages.append(args),
    )
    window.prepare_spike_detection()

    window.approve_and_run()

    assert calls == []
    assert messages
    assert window._prepared_execution is not None


def test_signal_kind_becomes_confirmed_only_after_researcher_yes(window, tmp_path, monkeypatch):
    path = tmp_path / "unknown.npy"
    np.save(path, np.arange(20, dtype=np.float64))
    prepared = prepare_dataset(path)
    assert prepared is not None
    assert prepared.exploration.summary.metadata["signal_kind"].status.value == "unknown"

    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )
    confirmed = window._confirm_raw_voltage_if_requested(prepared)

    signal_kind = confirmed.exploration.summary.metadata["signal_kind"]
    assert signal_kind.value == "raw_voltage"
    assert signal_kind.status.value == "confirmed"
    assert confirmed.exploration.summary.metadata["sampling_rate_hz"].status.value == "unknown"
    assert confirmed.exploration.summary.metadata["units"].status.value == "unknown"


def test_negative_detection_approval_execution_persistence_overlay_and_selection(
    window, tmp_path, monkeypatch, app
):
    values = np.random.default_rng(23).normal(0, 0.1, 120)
    values[32] = -8.0
    values[82] = -9.0
    _install_dataset(window, values, tmp_path, fs=2000)
    window.navigation_state.zoom(20, 70)
    window._redraw_navigation_view()
    _select_project(window, tmp_path / "project")
    window.prepare_spike_detection()
    _approve_dialogs(monkeypatch)

    window.approve_and_run()
    assert window._execution_in_flight
    assert window._current_record is None
    _wait_for_worker(window, app)

    record = window._current_record
    assert record is not None and record.status.value == "succeeded"
    assert record.operation_id == NEGATIVE_ID
    assert record.provenance.approved_by == "Test Researcher"
    assert [item["original_sample_index"] for item in window._current_events] == [32]
    assert window.event_table.rowCount() == 1
    scatter = window._event_scatters[window.figure.axes[0]]
    np.testing.assert_allclose(scatter.get_offsets(), [[32 / 2000, values[32]]])
    saved = window.store.load_execution_history()
    assert len(saved) == 1
    assert saved[0]["provenance"]["approval_id"] == record.provenance.approval_id
    assert "output" in saved[0]

    window.event_table.cellClicked.emit(0, 0)
    assert "Original sample index: 32" in window.event_details.toPlainText()
    assert "0.016 s" in window.event_details.toPlainText()
    assert window.navigation_state.visible_window.start <= 32
    assert window.navigation_state.visible_window.stop > 32


def test_reopen_project_restores_detector_history_without_reexecution(
    window, tmp_path, monkeypatch, app
):
    values = np.zeros(120, dtype=np.float64)
    values[45] = -8.0
    _install_dataset(window, values, tmp_path, fs=2000)
    project_path = tmp_path / "recoverable-project"
    _select_project(window, project_path)
    window.store.save_profile(window.active_dataset.profile)
    window.prepare_spike_detection()
    _approve_dialogs(monkeypatch)
    window.approve_and_run()
    _wait_for_worker(window, app)
    original = window._current_record
    assert original is not None and original.status.value == "succeeded"
    reference = original.data_references[0]
    saved_execution_id = original.execution_id

    reopened = MainWindow()
    reopened.open_project(str(project_path))
    app.processEvents()
    try:
        assert reopened.active_dataset is None
        assert reopened._current_record is not None, reopened.operation_review.toPlainText()
        assert reopened._current_record.execution_id == saved_execution_id
        assert reopened._current_events[0]["original_sample_index"] == 45
        assert reopened.waveform_source_box.findData(saved_execution_id) >= 0
        assert "no analyses were rerun" in reopened.analysis_state_label.text()
        with pytest.raises(DatasetAccessError, match="unknown or stale dataset handle"):
            reopened.dataset_store.resolve(reference)
    finally:
        reopened.close()


def test_failed_project_preflight_preserves_current_project(window, tmp_path, monkeypatch):
    current = ProjectStore(tmp_path / "current")
    window.store = current
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "project.json").write_text("{broken", encoding="utf-8")
    errors = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "critical",
        lambda *args, **kwargs: errors.append(args[2]),
    )

    window.open_project(str(invalid))

    assert window.store is current
    assert errors and "Project manifest is invalid" in errors[0]


def test_two_projects_can_be_opened_sequentially(window, tmp_path):
    first = ProjectStore(tmp_path / "first")
    second = ProjectStore(tmp_path / "second")

    window.open_project(str(first.root))
    first_id = window.store.project_id
    window.open_project(str(second.root))

    assert first_id != window.store.project_id
    assert window.store.project_id == second.project_id
    assert window.active_dataset is None


def test_opening_project_invalidates_unexecuted_prepared_operation(window, tmp_path):
    _install_dataset(window, np.zeros(40, dtype=np.float64), tmp_path)
    _select_project(window, tmp_path / "before")
    window.prepare_spike_detection()
    assert window._prepared_execution is not None
    prepared_id = window._prepared_execution.request_fingerprint
    target = ProjectStore(tmp_path / "after")

    window.open_project(str(target.root))

    assert window._prepared_execution is None
    assert window.active_dataset is None
    assert target.load_execution_history() == []
    assert prepared_id


def test_source_reconnect_requires_confirmation_and_preserves_dataset_identity(
    window, tmp_path, monkeypatch
):
    original_path = tmp_path / "original.npy"
    moved_path = tmp_path / "moved.npy"
    values = np.arange(80, dtype=np.float64)
    np.save(original_path, values)
    np.save(moved_path, values)
    saved = prepare_dataset(original_path)
    candidate = prepare_dataset(moved_path)
    store = ProjectStore(tmp_path / "project")
    store.save_profile(saved.profile)
    window.store = store
    saved_profile = saved.profile
    saved.close()
    choices = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *args, **kwargs: choices.append(args[2]) or QMessageBox.Yes,
    )

    reconnected = window._resolve_project_source(candidate)

    assert reconnected is not None
    assert choices and "bounded partial fingerprint" in choices[0]
    assert reconnected.profile.dataset_id == saved_profile.dataset_id
    assert reconnected.profile.source_path == str(moved_path.resolve())
    assert reconnected.profile.unknown_fields == saved_profile.unknown_fields
    assert json.loads(store.source_resolution_path.read_text(encoding="utf-8"))["status"] == (
        "researcher_confirmed_partial_match"
    )
    reconnected.close()


@pytest.mark.parametrize(
    ("mode", "threshold", "noise", "expected_sample", "expected_polarity"),
    [
        ("absolute", "2", "1", 25, "positive"),
        ("absolute", "-2", "1", 70, "negative"),
        ("noise_multiplier", "2", "", 25, "positive"),
        ("noise_multiplier", "-2", "", 70, "negative"),
        ("absolute", "0", "1", None, "none"),
        ("noise_multiplier", "0", "", None, "none"),
    ],
)
def test_signed_peak_modes_polarity_zero_and_filter_warning(
    window, tmp_path, monkeypatch, app, mode, threshold, noise, expected_sample, expected_polarity
):
    _install_dataset(window, _zero_noise_signed_signal(), tmp_path)
    _select_project(window, tmp_path / f"project-{mode}-{threshold}")
    window.detector_box.setCurrentIndex(1)
    window.signed_mode_box.setCurrentIndex(
        0 if mode == "absolute" else 1
    )
    window.signed_threshold.setText(threshold)
    window.noise_override.setText(noise)
    window.prepare_spike_detection()

    prepared = window._prepared_execution
    assert prepared is not None
    assert prepared.parameters.threshold_mode == mode
    assert "Filtering information is missing" in window.operation_review.toPlainText()
    assert window._current_record is None  # preparation did not execute
    _approve_dialogs(monkeypatch)
    window.approve_and_run()
    _wait_for_worker(window, app)

    output = window._current_record.output
    assert output is not None
    assert output.polarity == expected_polarity
    if expected_sample is None:
        assert output.original_sample_indices == ()
    else:
        assert expected_sample in output.original_sample_indices
    assert "Filtering information is missing" in "\n".join(window._current_record.warnings)
    assert output.sampling_rate_hz is None
    assert output.amplitude_units is None
    assert output.noise_source == (
        "overridden" if mode == "absolute" and noise else "estimated"
    )
    if expected_sample is None:
        assert "No events detected" in window.operation_review.toPlainText()


def test_navigation_never_reruns_detection_and_old_approval_is_invalidated(
    window, tmp_path, monkeypatch, app
):
    values = np.random.default_rng(29).normal(0, 0.1, 100)
    values[40] = -8
    _install_dataset(window, values, tmp_path)
    _select_project(window, tmp_path / "project")
    calls = []
    original_execute = window.operation_runner.execute

    def counted(approved):
        calls.append(approved)
        return original_execute(approved)

    monkeypatch.setattr(window.operation_runner, "execute", counted)
    window.prepare_spike_detection()
    _approve_dialogs(monkeypatch)
    window.approve_and_run()
    _wait_for_worker(window, app)
    assert len(calls) == 1

    window.navigation_state.zoom(30, 60)
    window._redraw_navigation_view()
    assert len(calls) == 1

    window.prepare_spike_detection()
    window.figure.axes[0].set_xlim(20, 70)
    assert window._prepared_execution is None
    assert len(calls) == 1


def test_dataset_replacement_marks_prior_result_stale(window, tmp_path):
    first = np.zeros(50)
    _install_dataset(window, first, tmp_path)
    old_registration = window.active_registration
    window.prepare_spike_detection()
    replacement = _dataset(np.ones(40), tmp_path / "second.npy")

    window._commit_dataset(replacement)

    assert window.active_registration.session_handle != old_registration.session_handle
    with pytest.raises(DatasetAccessError, match="stale"):
        window.dataset_store.resolve(
            old_registration.reference((0,), SampleInterval(start=0, stop=10))
        )
    assert window._prepared_execution is None


def _write_mat73(path: Path, logical: np.ndarray) -> None:
    with h5py.File(path, "w", userblock_size=512) as h5:
        signal = np.asarray(logical)
        h5.create_dataset("signal", data=signal, chunks=True)
    with path.open("r+b") as stream:
        stream.write(b"MATLAB 7.3 MAT-file".ljust(116, b" ") + b"\0" * 8 + b"\x00\x02IM")


def test_lazy_matlab_detection_reads_only_approved_window(window, tmp_path, monkeypatch, app):
    path = tmp_path / "lazy.mat"
    values = np.random.default_rng(31).normal(0, 0.1, 200)
    values[123] = -8
    _write_mat73(path, values)
    source = open_matlab_v73(path, "signal")
    profile = DatasetProfile.from_array(path, source)
    profile = profile.model_copy(
        update={
            "metadata": profile.metadata.model_copy(update={"signal_kind": "raw_voltage"}),
            "confirmed_fields": ["signal_kind"],
            "unknown_fields": ["sampling_rate_hz", "units", "filtering_description"],
        }
    )
    prepared = PreparedDataset(path, source, profile, DatasetExploration(source, profile))
    window._commit_dataset(prepared)
    window.navigation_state.zoom(100, 150)
    window._redraw_navigation_view()
    reads = []
    original = source.read_window

    def observed(start, stop, **kwargs):
        reads.append((start, stop, kwargs))
        return original(start, stop, **kwargs)

    monkeypatch.setattr(source, "read_window", observed)
    _select_project(window, tmp_path / "lazy-project")
    window.prepare_spike_detection()
    _approve_dialogs(monkeypatch)
    window.approve_and_run()
    _wait_for_worker(window, app)

    assert reads
    assert all((start, stop) == (100, 150) for start, stop, _ in reads)
    assert window._current_record.output.original_sample_indices == (123,)


def test_confirmed_filtering_metadata_is_preserved_without_warning(
    window, tmp_path, monkeypatch, app
):
    _install_configured_dataset(
        window,
        _zero_noise_signed_signal(),
        tmp_path,
        filtering_description="explicit test setup: 300 Hz high-pass",
    )
    _select_project(window, tmp_path / "project")
    window.detector_box.setCurrentIndex(1)
    window.signed_threshold.setText("2")
    window.noise_override.setText("1")
    window.prepare_spike_detection()
    assert "Filtering information is missing" not in window.operation_review.toPlainText()
    _approve_dialogs(monkeypatch)
    window.approve_and_run()
    _wait_for_worker(window, app)

    output = window._current_record.output
    assert output.filtering_description == "explicit test setup: 300 Hz high-pass"
    assert output.filtering_status.value == "confirmed"
    assert output.algorithm_provenance.filtering_warning is False


def test_worker_failure_is_reported_and_saved(window, tmp_path, monkeypatch, app):
    values = np.random.default_rng(37).normal(0, 0.1, 80)
    _install_dataset(window, values, tmp_path)
    _select_project(window, tmp_path / "project")
    window.prepare_spike_detection()

    def fail_resolve(_reference):
        raise RuntimeError("controlled worker failure")

    monkeypatch.setattr(window.dataset_store, "resolve", fail_resolve)
    _approve_dialogs(monkeypatch)
    monkeypatch.setattr(main_window_module.QMessageBox, "critical", lambda *args: None)
    window.approve_and_run()
    _wait_for_worker(window, app)

    assert window._current_record.status.value == "failed"
    assert "controlled worker failure" in window._current_record.error.message
    assert window.store.load_execution_history()[0]["status"] == "failed"


def test_running_worker_prevents_close_until_lazy_source_is_safe(
    window, tmp_path, monkeypatch, app
):
    values = np.random.default_rng(41).normal(0, 0.1, 80)
    values[35] = -8
    _install_dataset(window, values, tmp_path)
    _select_project(window, tmp_path / "project")
    window.prepare_spike_detection()
    entered = threading.Event()
    release = threading.Event()
    original_execute = window.operation_runner.execute

    def blocked_execute(approved):
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test worker release timed out")
        return original_execute(approved)

    monkeypatch.setattr(window.operation_runner, "execute", blocked_execute)
    _approve_dialogs(monkeypatch)
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args: None)
    window.approve_and_run()
    assert entered.wait(timeout=3)

    close_event = QCloseEvent()
    window.closeEvent(close_event)
    assert not close_event.isAccepted()
    assert window.active_dataset is not None

    release.set()
    _wait_for_worker(window, app)
    assert window.active_dataset is not None


def test_waveform_view(
    window, tmp_path, monkeypatch, app
):
    values = np.tile(np.asarray([0.01, -0.01]), 75).astype(np.float64)
    values[2] = -8.0
    values[50] = -9.0
    values[100] = -10.0
    _install_dataset(window, values, tmp_path, fs=1000)
    _select_project(window, tmp_path / "waveform-project")
    window.prepare_spike_detection()
    _approve_dialogs(monkeypatch)
    window.approve_and_run()
    _wait_for_worker(window, app)
    detector_record = window._current_record
    assert detector_record.status.value == "succeeded"
    assert [event["original_sample_index"] for event in window._current_events] == [2, 50, 100]

    # A newer detector result may be current while the researcher chooses an
    # earlier compatible execution for waveform extraction.
    window.threshold.setText("-6")
    window.prepare_spike_detection()
    window.approve_and_run()
    _wait_for_worker(window, app)
    assert window._current_record.execution_id != detector_record.execution_id

    window.waveform_pre_samples.setText("5")
    window.waveform_post_samples.setText("7")
    source_index = window.waveform_source_box.findData(detector_record.execution_id)
    window.waveform_source_box.setCurrentIndex(source_index)
    window.prepare_waveform_extraction()
    prepared = window._prepared_execution
    assert prepared is not None
    assert prepared.input_data.detections.events[0].event_id == f"{detector_record.execution_id}:0"
    assert "waveform length: 13" in window.operation_review.toPlainText()
    _approve_dialogs(monkeypatch)
    window.approve_and_run()
    _wait_for_worker(window, app)
    _wait_for_view(window, app, "waveform")

    extraction = window._current_waveform_record
    output = extraction.output
    assert output.accepted_event_count == 2
    assert output.excluded_event_count == 1
    assert output.waveform_shape == (2, 13)
    assert window.waveform_excluded_table.rowCount() == 1
    assert window.waveform_excluded_table.item(0, 0).text() == f"{detector_record.execution_id}:0"
    assert "deterministic subset" in window.waveform_display_label.text()
    assert window.waveform_figure.axes[0].get_xlabel() == "Time relative to detector event (s)"
    with window.store.artifacts.open(output.artifact_reference) as artifact:
        assert artifact.waveforms.shape == (2, 13)

    recovered = MainWindow()
    recovered.open_project(str(window.store.root))
    app.processEvents()
    try:
        assert recovered.waveform_saved_box.count() == 2
        recovered.waveform_saved_box.setCurrentIndex(1)
        recovered.open_saved_waveform_result()
        _wait_for_view(recovered, app, "waveform")
        assert recovered._current_waveform_record.execution_id == extraction.execution_id
        assert "source navigation is disabled" in recovered.waveform_display_label.text()
        assert recovered.waveform_figure.axes
    finally:
        recovered.close()


    # Selecting a waveform identifies the detector event and navigates the signal view.
    second_index = window.waveform_event_box.findData(f"{detector_record.execution_id}:2")
    window.waveform_event_box.setCurrentIndex(second_index)
    _wait_for_view(window, app, "waveform")
    assert window._current_record is detector_record
    assert window.event_table.currentRow() == 2
    assert window.navigation_state.visible_window.start <= 100
    assert window.navigation_state.visible_window.stop > 100


def test_isi_view(
    window, tmp_path, monkeypatch, app
):
    values = np.tile(np.asarray([0.01, -0.01]), 75).astype(np.float64)
    values[25] = -8.0
    values[50] = -9.0
    values[75] = -10.0
    _install_dataset(window, values, tmp_path)
    _select_project(window, tmp_path / "isi-project")
    window.prepare_spike_detection()
    _approve_dialogs(monkeypatch)
    window.approve_and_run()
    _wait_for_worker(window, app)
    detector_record = window._current_record
    assert detector_record.status.value == "succeeded"
    assert [event["original_sample_index"] for event in window._current_events] == [25, 50, 75]

    source_index = window.isi_source_box.findData(detector_record.execution_id)
    window.isi_source_box.setCurrentIndex(source_index)
    window.isi_polarity_box.setCurrentIndex(window.isi_polarity_box.findData("separate"))
    window.isi_threshold_enabled.setChecked(True)
    window.isi_threshold.setText("25")
    window.isi_threshold_units.setCurrentIndex(window.isi_threshold_units.findData("samples"))
    window.prepare_isi_analysis()

    prepared = window._prepared_execution
    assert prepared is not None
    assert (
        prepared.input_data.events.dataset.dataset_id
        == detector_record.data_references[0].dataset.dataset_id
    )
    review = window.operation_review.toPlainText().lower()
    assert "polarity grouping: separate" in review
    assert "25" in review and "samples" in review
    _approve_dialogs(monkeypatch)
    window.approve_and_run()
    _wait_for_worker(window, app)
    _wait_for_view(window, app, "isi")

    isi_record = window._current_isi_record
    output = isi_record.output
    assert output.detector_execution_id == detector_record.execution_id
    assert output.polarity_mode == "separate"
    assert output.interval_time_unit == "unavailable"
    summary = output.summaries[0]
    assert summary.interval_count == 2
    assert summary.mean_isi_samples == 25
    assert summary.intervals_below_threshold_count == 0  # Strict comparison: equality is not below.
    assert summary.short_interval_fraction == 0
    assert window.isi_interval_table.rowCount() == 2
    assert window.isi_figure.axes[0].get_xlabel() == "Inter-spike interval (samples)"
    assert "no events were modified" not in window.isi_display_label.text().lower()
    assert "not confirmed refractory violations" in window.isi_display_label.text().lower()

    first_interval = window._isi_page_rows[0]
    assert first_interval["starting_sample_index"] == 25
    assert first_interval["ending_sample_index"] == 50
    window._select_isi_interval(0, 0)
    assert window.navigation_state.visible_window.start <= 25
    assert window.navigation_state.visible_window.stop > 50
    assert "no events were modified" in window.isi_display_label.text().lower()

    recovered = MainWindow()
    recovered.open_project(str(window.store.root))
    app.processEvents()
    try:
        assert recovered.isi_saved_box.count() == 2
        recovered.isi_saved_box.setCurrentIndex(1)
        recovered.open_saved_isi_result()
        _wait_for_view(recovered, app, "isi")
        assert recovered._current_isi_record.execution_id == isi_record.execution_id
        assert recovered.isi_interval_table.rowCount() == 2
        assert "source navigation is disabled" in recovered.isi_display_label.text()
    finally:
        recovered.close()


def test_isi_millisecond_threshold_requires_confirmed_sampling_rate(
    window, tmp_path, monkeypatch, app
):
    values = np.tile(np.asarray([0.01, -0.01]), 50).astype(np.float64)
    values[20] = -8.0
    values[40] = -9.0
    _install_dataset(window, values, tmp_path, fs=None)
    _select_project(window, tmp_path / "isi-unknown-rate-project")
    window.prepare_spike_detection()
    _approve_dialogs(monkeypatch)
    window.approve_and_run()
    _wait_for_worker(window, app)
    detector_record = window._current_record

    window.isi_source_box.setCurrentIndex(
        window.isi_source_box.findData(detector_record.execution_id)
    )
    window.isi_polarity_box.setCurrentIndex(window.isi_polarity_box.findData("separate"))
    window.isi_threshold_enabled.setChecked(True)
    window.isi_threshold.setText("2.5")
    window.isi_threshold_units.setCurrentIndex(
        window.isi_threshold_units.findData("milliseconds")
    )
    window.prepare_isi_analysis()

    assert window._prepared_execution is None
    assert "confirmed sampling rate" in window.operation_review.toPlainText().lower()
    assert window._current_isi_record is None


def test_mua_overlay(
    window, tmp_path, monkeypatch, app
):
    values = np.tile(np.asarray([0.1, -0.1]), 50).astype(np.float64)
    values[30] = 2.0
    values[50] = -2.0
    _install_dataset(window, values, tmp_path, fs=1000)
    _select_project(window, tmp_path / "p")
    window.mua_channel_box.setCurrentIndex(window.mua_channel_box.findData(0))
    window.mua_stdmin.setText("1")
    window.mua_bandwidth.setText("0.003")
    window.mua_mode_box.setCurrentIndex(window.mua_mode_box.findData("rel"))
    window.mua_num_points.setText("100")
    window.prepare_calc_mua()

    prepared = window._prepared_execution
    assert prepared is not None
    assert prepared.input_data.signal.channels == (0,)
    assert prepared.input_data.signal.sample_interval.start == 0
    assert prepared.input_data.signal.sample_interval.stop == 100
    review = window.operation_review.toPlainText()
    assert "calcmua_verFi.m" in review
    assert "median(abs(signal))/0.6745" in review
    assert "does not filter" in review
    assert "muawin: 0.003 s" in review

    _approve_dialogs(monkeypatch)
    window.approve_and_run()
    _wait_for_worker(window, app)
    _wait_for_view(window, app, "mua")

    record = window._current_mua_record
    output = record.output
    assert record.status.value == "succeeded"
    assert output.event_count == 1
    assert output.mua_units == "s^-1"
    assert output.sample_interval.start == 0
    assert output.sample_interval.stop == 100
    assert len(window.mua_figure.axes) == 2
    assert window.mua_figure.axes[0].lines[0].get_label() == "Source signal"
    assert window.mua_figure.axes[1].lines[0].get_label() == "calcMUA trace"
    assert "filtering is not confirmed" in window.mua_display_label.text()
    with window.store.mua_artifacts.open(output.artifact_reference) as artifact:
        assert artifact.mua.shape == (100,)
        assert artifact.spike_sample_indices.tolist() == [30]

    execution_count = len(window._execution_records)
    window._on_mua_plot_click(type("MouseEvent", (), {"xdata": 0.03})())
    assert window.navigation_state.visible_window.start <= 30
    assert window.navigation_state.visible_window.stop > 30
    assert window.workspace_tabs.currentIndex() == 0
    assert len(window._execution_records) == execution_count


def test_mua_batch(window, tmp_path, monkeypatch, app):
    time = np.arange(160, dtype=np.float64)
    values = np.column_stack(
        (
            np.where(time % 16 == 8, -2.0, 0.1),
            np.where(time % 20 == 10, -3.0, 0.2),
        )
    )
    _install_dataset(
        window,
        values,
        tmp_path,
        fs=1000,
        channel_axis=1,
        units="mV",
        filtering_description="confirmed synthetic bandpass preprocessing",
    )
    _select_project(window, tmp_path / "p")
    for index in range(2):
        window.mua_calculation_channels.item(index).setCheckState(Qt.Checked)
    window.mua_stdmin.setText("-1")
    window.mua_bandwidth.setText("0.004")
    window.mua_mode_box.setCurrentIndex(window.mua_mode_box.findData("rel"))
    window.mua_num_points.setText("160")
    window.prepare_calc_mua()
    assert window._prepared_mua_batch is not None
    assert [item.input_data.signal.channels for item in window._prepared_mua_batch] == [(0,), (1,)]
    assert "separately validated, approved, executed" in window.operation_review.toPlainText()

    _approve_dialogs(monkeypatch)
    window.approve_and_run()
    _wait_for_worker(window, app)
    _wait_for_view(window, app, "mua")
    records = [
        record for record in window._mua_records.values()
        if record.operation_id == "neuroscope.mua.calc-mua"
    ]
    assert len(records) == 2, f"state={window.analysis_state_label.text()} review={window.operation_review.toPlainText()}"
    assert all(record.status.value == "succeeded" for record in records)
    batch_refs = {record.context.reproducibility.configuration_reference for record in records}
    assert len(batch_refs) == 1
    assert window.mua_display_results.count() == 2
    assert window.mua_average_inputs.count() == 2
    assert len(window._mua_selection_views) == 2
    assert len(window.mua_figure.axes[0].lines) == 2

    execution_count = len(window._execution_records)
    window.mua_display_results.item(0).setCheckState(Qt.Unchecked)
    _wait_for_view(window, app, "mua")
    assert len(window._execution_records) == execution_count
    assert sum(
        window.mua_display_results.item(i).checkState() == Qt.Checked
        for i in range(window.mua_display_results.count())
    ) == 1

    for index in range(window.mua_average_inputs.count()):
        window.mua_average_inputs.item(index).setCheckState(Qt.Checked)
    window.prepare_mua_average()
    assert window._prepared_execution is not None
    assert window._prepared_execution.operation_id == "neuroscope.mua.average-traces"
    assert "Selected artifact count: 2" in window.operation_review.toPlainText()
    average_prepared = window._prepared_execution
    window.mua_display_results.item(0).setCheckState(Qt.Checked)
    assert window._prepared_execution is average_prepared
    window.mua_average_inputs.item(0).setCheckState(Qt.Unchecked)
    assert window._prepared_execution is None
    window.mua_average_inputs.item(0).setCheckState(Qt.Checked)
    window.prepare_mua_average()
    assert window._prepared_execution is not None
    _approve_dialogs(monkeypatch)
    window.approve_and_run()
    _wait_for_worker(window, app)
    _wait_for_view(window, app, "mua")
    averages = [
        record for record in window._mua_records.values()
        if record.operation_id == "neuroscope.mua.average-traces"
    ]
    assert len(averages) == 1
    assert averages[0].status.value == "succeeded"
    assert averages[0].output.contributing_channels == 2
    assert len(averages[0].output.source_artifact_references) == 2
    with window.store.mua_artifacts.open(averages[0].output.artifact_reference) as artifact:
        assert artifact.mua.shape == (160,)
        assert artifact.manifest["artifact_type"] == "averaged_mua_trace"
    window._refresh_mua_project_results()
    assert window.mua_display_results.count() == 3, window.operation_review.toPlainText()
    assert window.mua_average_inputs.count() == 2
    assert averages[0].execution_id in window._mua_provenance_snapshots
    assert len(window._mua_input_references) == 2

    recovered = MainWindow()
    recovered.open_project(str(window.store.root))
    app.processEvents()
    try:
        assert recovered.mua_display_results.count() == 3
        assert recovered.mua_average_inputs.count() == 2
        assert all(
            recovered.mua_display_results.item(i).checkState() == Qt.Unchecked
            for i in range(recovered.mua_display_results.count())
        )
        with recovered.store.mua_artifacts.open(
            next(
                output.artifact_reference
                for output in recovered._mua_outputs.values()
                if getattr(output, "contributing_channels", None) == 2
            )
        ) as restored_artifact:
            assert restored_artifact.mua.shape == (160,)
    finally:
        recovered.close()


def test_pipeline_explorer_is_integrated_with_separate_review_and_approval(window):
    tab_names = [window.workspace_tabs.tabText(index) for index in range(window.workspace_tabs.count())]
    assert "Pipelines" in tab_names
    assert not window.pipeline_json_editor.isReadOnly()
    assert window.pipeline_prepare_button.text() == "Prepare Next Node"
    assert window.pipeline_approve_button.text() == "Approve & Execute Prepared Node"
    assert window.pipeline_service.runner is window.operation_runner


def test_pipeline_gui_can_create_save_and_reopen_without_running_operations(
    window, tmp_path, monkeypatch
):
    _select_project(window, tmp_path / "pipeline-project")
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("Review-only draft", True),
    )
    window._new_pipeline_draft()
    first_operation = window.operation_registry.list_operations()[0]
    first_operation_label = (
        f"{first_operation.operation_id} v{first_operation.version} — {first_operation.name}"
    )
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getItem",
        lambda *_args, **_kwargs: (first_operation_label, True),
    )
    window._pipeline_add_operation_node()
    draft = main_window_module.PipelineDefinition.model_validate_json(
        window.pipeline_json_editor.toPlainText(), strict=True
    )
    assert len(draft.nodes) == 1
    assert draft.nodes[0].parameters == {}
    window._save_pipeline_ui()
    saved = window.store.load_pipeline(draft.pipeline_id)
    assert saved == draft

    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getItem",
        lambda *_args, **_kwargs: (
            f"{saved.name} — r{saved.revision} ({saved.pipeline_id})",
            True,
        ),
    )
    window._open_saved_pipeline()
    assert "did not execute" in window.pipeline_status.toPlainText()
    assert window._pipeline_run_state is None

