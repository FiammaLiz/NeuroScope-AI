import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from scipy.io import savemat

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
    instance.close()


def _open(window, monkeypatch, path):
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *args, **kwargs: main_window_module.QMessageBox.No,
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(path), ""),
    )
    window.load_dataset()


def _visible_state(window):
    axes = window.figure.axes
    plot_title = axes[0].get_title() if axes else None
    channels = tuple(
        window.channel_box.itemText(index)
        for index in range(window.channel_box.count())
    )
    plotted_values = tuple(
        tuple(axis.lines[0].get_ydata().tolist()) if axis.lines else None
        for axis in axes
    )
    return (
        window.active_dataset,
        window.project_label.text(),
        window.profile_view.toPlainText(),
        channels,
        plot_title,
        plotted_values,
        (
            window.navigation_state.selected_channels,
            window.navigation_state.visible_window,
        )
        if window.navigation_state is not None
        else None,
    )


def test_valid_dataset_commits_and_opening_does_not_create_project_store(
    window, monkeypatch, tmp_path
):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "recording.npy"
    np.save(path, values)
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getItem",
        lambda *args, **kwargs: ("Axis 0 (rows are channels)", True),
    )

    _open(window, monkeypatch, path)

    assert window.active_dataset is not None
    assert window.active_dataset.profile.metadata.channel_axis == 0
    assert window.project_label.text() == "recording.npy"
    assert tuple(window.channel_box.itemText(i) for i in range(3)) == (
        "Channel 1",
        "Channel 2",
        "Channel 3",
    )
    assert "Channel orientation: channels on axis 0 [CONFIRMED]" in window.profile_view.toPlainText()
    assert window.figure.axes[-1].get_xlabel() == "Sample index"
    assert not (tmp_path / ".recording_neuroscope").exists()
    assert window.store is None


def test_multichannel_selection_updates_stacked_plot_without_changing_time(
    window, monkeypatch, tmp_path
):
    values = np.arange(30, dtype=np.float64).reshape(3, 10)
    path = tmp_path / "channels.npy"
    np.save(path, values)
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getItem",
        lambda *args, **kwargs: ("Axis 0 (rows are channels)", True),
    )
    _open(window, monkeypatch, path)
    assert len(window.figure.axes) == 3
    window.navigation_state.zoom(2, 7)
    window._redraw_navigation_view()

    window.display_channel_list.item(1).setCheckState(main_window_module.Qt.Unchecked)

    assert window.navigation_state.selected_channels == (0, 2)
    assert (window.navigation_state.visible_window.start,
            window.navigation_state.visible_window.stop) == (2, 7)
    assert len(window.figure.axes) == 2
    assert [axis.get_title() for axis in window.figure.axes] == [
        "Channel 0 (original index)",
        "Channel 2 (original index)",
    ]
    np.testing.assert_array_equal(
        window.figure.axes[0].lines[0].get_ydata(), values[0, 2:7]
    )
    np.testing.assert_array_equal(
        window.figure.axes[1].lines[0].get_ydata(), values[2, 2:7]
    )


def test_clear_channels_shows_empty_state_and_reset_restores_full_recording(
    window, monkeypatch, tmp_path
):
    path = tmp_path / "channels.npy"
    np.save(path, np.arange(30).reshape(3, 10))
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getItem",
        lambda *args, **kwargs: ("Axis 0 (rows are channels)", True),
    )
    _open(window, monkeypatch, path)
    window.navigation_state.zoom(3, 8)
    window._redraw_navigation_view()
    window.clear_channels_button.click()

    assert window.navigation_state.selected_channels == ()
    assert "No channels selected" in window.figure.axes[0].texts[0].get_text()

    window.select_all_channels_button.click()
    window.figure.axes[0].set_xlim(5, 8)
    assert (window.navigation_state.visible_window.start,
            window.navigation_state.visible_window.stop) == (5, 8)
    window.navigation_toolbar.home()

    assert (window.navigation_state.visible_window.start,
            window.navigation_state.visible_window.stop) == (0, 10)
    assert window.figure.axes[0].get_xlim() == (0.0, 10.0)


def test_sample_index_navigation_and_confirmed_seconds_axis(
    window, monkeypatch, tmp_path
):
    path = tmp_path / "unknown_rate.npy"
    np.save(path, np.arange(8))
    _open(window, monkeypatch, path)
    assert window.figure.axes[0].get_xlabel() == "Sample index"
    window.figure.axes[0].set_xlim(2, 5)
    assert (window.navigation_state.visible_window.start,
            window.navigation_state.visible_window.stop) == (2, 5)

    prepared = window.active_dataset
    metadata = prepared.profile.metadata.model_copy(
        update={"sampling_rate_hz": 4.0}
    )
    profile = prepared.profile.model_copy(
        update={
            "metadata": metadata,
            "confirmed_fields": ["sampling_rate_hz"],
            "unknown_fields": ["units", "signal_kind"],
        }
    )
    from neuroscope.dataset_loading import PreparedDataset
    from neuroscope.exploration import DatasetExploration

    confirmed = PreparedDataset(
        prepared.path,
        prepared.array,
        profile,
        DatasetExploration(prepared.array, profile),
    )
    window._commit_dataset(confirmed)
    axis = window.figure.axes[0]
    assert axis.get_xlabel() == "Time relative to sample 0 (s)"
    axis.set_xlim(0.5, 1.25)
    assert (window.navigation_state.visible_window.start,
            window.navigation_state.visible_window.stop) == (2, 5)


def test_new_dataset_resets_navigation_state_and_failed_load_preserves_it(
    window, monkeypatch, tmp_path
):
    first_path = tmp_path / "first.npy"
    second_path = tmp_path / "second.npy"
    np.save(first_path, np.arange(30).reshape(3, 10))
    np.save(second_path, np.arange(12).reshape(2, 6))
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getItem",
        lambda *args, **kwargs: ("Axis 0 (rows are channels)", True),
    )
    _open(window, monkeypatch, first_path)
    window.navigation_state.zoom(2, 8)
    window.display_channel_list.item(1).setCheckState(main_window_module.Qt.Unchecked)
    before = _visible_state(window)
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getItem",
        lambda *args, **kwargs: ("", False),
    )
    _open(window, monkeypatch, second_path)
    assert _visible_state(window) == before

    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getItem",
        lambda *args, **kwargs: ("Axis 0 (rows are channels)", True),
    )
    _open(window, monkeypatch, second_path)
    assert window.navigation_state.selected_channels == (0, 1)
    assert (window.navigation_state.visible_window.start,
            window.navigation_state.visible_window.stop) == (0, 6)
    assert window._render_cache_key[0] == id(window.active_dataset)
    np.testing.assert_array_equal(
        window.figure.axes[0].lines[0].get_ydata(), np.arange(6)
    )


def test_cursor_readout_uses_original_sample_and_synchronizes_channels(
    window, monkeypatch, tmp_path
):
    values = np.vstack((np.arange(1000), 1000 + np.arange(1000), 2000 + np.arange(1000)))
    values[2, 567] = 99_999
    path = tmp_path / "long.npy"
    np.save(path, values)
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getItem",
        lambda *args, **kwargs: ("Axis 0 (rows are channels)", True),
    )
    _open(window, monkeypatch, path)
    window.display_channel_list.item(0).setCheckState(main_window_module.Qt.Unchecked)
    window.display_channel_list.item(1).setCheckState(main_window_module.Qt.Unchecked)

    selected_axis = window.figure.axes[0]
    window._on_cursor_motion(
        SimpleNamespace(inaxes=selected_axis, xdata=567.0)
    )

    message = window.statusBar().currentMessage()
    assert "Channel 2" in message
    assert "sample 567" in message
    assert "99999" in message
    assert "t=" not in message
    assert selected_axis.lines[1].get_visible()
    assert selected_axis.lines[1].get_xdata()[0] == 567

    window._on_cursor_motion(SimpleNamespace(inaxes=None, xdata=None))
    assert window.statusBar().currentMessage() == ""
    assert not selected_axis.lines[1].get_visible()


def test_cursor_uses_confirmed_seconds_but_original_value_when_reduced(
    window, monkeypatch, tmp_path
):
    values = np.zeros(5000, dtype=np.float64)
    values[4321] = 12.5
    path = tmp_path / "spike.npy"
    np.save(path, values)
    _open(window, monkeypatch, path)
    prepared = window.active_dataset
    metadata = prepared.profile.metadata.model_copy(
        update={"sampling_rate_hz": 1000.0}
    )
    profile = prepared.profile.model_copy(
        update={
            "metadata": metadata,
            "confirmed_fields": ["sampling_rate_hz"],
            "unknown_fields": ["units", "signal_kind"],
        }
    )
    from neuroscope.dataset_loading import PreparedDataset
    from neuroscope.exploration import DatasetExploration

    window._commit_dataset(
        PreparedDataset(
            prepared.path,
            prepared.array,
            profile,
            DatasetExploration(prepared.array, profile),
        )
    )
    axis = window.figure.axes[0]
    assert len(axis.lines[0].get_xdata()) < len(values)
    assert "min/max envelope (display only)" in axis.get_title()
    window._on_cursor_motion(
        SimpleNamespace(inaxes=axis, xdata=4321 / 1000)
    )

    message = window.statusBar().currentMessage()
    assert "sample 4321" in message
    assert "value 12.5" in message
    assert "t=4.321 s" in message


def test_resize_and_zoom_adapt_render_representation_and_reuse_unchanged_cache(
    window, monkeypatch, tmp_path
):
    values = np.sin(np.arange(5000, dtype=np.float64) / 7)
    path = tmp_path / "long.npy"
    np.save(path, values)
    _open(window, monkeypatch, path)
    width = [100]
    monkeypatch.setattr(window, "_available_plot_width", lambda: width[0])
    window._redraw_navigation_view()
    first_cache = window._render_cache[0]
    initial_key = window._render_cache_key
    assert len(window.figure.axes[0].lines[0].get_xdata()) <= 200

    window._redraw_navigation_view()
    assert window._render_cache[0] is first_cache
    assert window._render_cache_key == initial_key

    width[0] = 20
    window._on_canvas_resize(None)
    assert window._render_cache_key != initial_key
    assert len(window.figure.axes[0].lines[0].get_xdata()) <= 40

    window.figure.axes[0].set_xlim(100, 130)
    assert (window.navigation_state.visible_window.start,
            window.navigation_state.visible_window.stop) == (100, 130)
    assert len(window.figure.axes[0].lines[0].get_xdata()) == 30


def test_cancelled_orientation_preserves_all_visible_state(
    window, monkeypatch, tmp_path
):
    first_path = tmp_path / "first.npy"
    second_path = tmp_path / "second.npy"
    np.save(first_path, np.arange(5))
    np.save(second_path, np.zeros((2, 5)))
    _open(window, monkeypatch, first_path)
    before = _visible_state(window)

    def cancel_orientation(*args, **kwargs):
        assert _visible_state(window) == before
        return "", False

    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getItem",
        cancel_orientation,
    )

    _open(window, monkeypatch, second_path)

    after = _visible_state(window)
    assert after == before


def test_cancelled_mat_variable_selection_preserves_all_visible_state(
    window, monkeypatch, tmp_path
):
    first_path = tmp_path / "first.npy"
    mat_path = tmp_path / "multiple.mat"
    np.save(first_path, np.arange(5))
    savemat(mat_path, {"left": np.arange(2), "right": np.arange(3)})
    _open(window, monkeypatch, first_path)
    before = _visible_state(window)

    def cancel_variable(*args, **kwargs):
        assert _visible_state(window) == before
        return "", False

    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        cancel_variable,
    )

    _open(window, monkeypatch, mat_path)

    assert _visible_state(window) == before


def test_loading_error_preserves_all_visible_state(window, monkeypatch, tmp_path):
    first_path = tmp_path / "first.npy"
    missing_path = tmp_path / "missing.npy"
    np.save(first_path, np.arange(5))
    _open(window, monkeypatch, first_path)
    before = _visible_state(window)
    errors = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "critical",
        lambda *args: errors.append(args[-1]),
    )

    _open(window, monkeypatch, missing_path)

    assert errors
    assert _visible_state(window) == before


@pytest.mark.parametrize("bad_data", [np.array([], dtype=float), np.array([1.0, np.nan])])
def test_invalid_dataset_preserves_all_visible_state(
    window, monkeypatch, tmp_path, bad_data
):
    first_path = tmp_path / "first.npy"
    invalid_path = tmp_path / "invalid.npy"
    np.save(first_path, np.arange(5))
    np.save(invalid_path, bad_data)
    _open(window, monkeypatch, first_path)
    before = _visible_state(window)
    monkeypatch.setattr(main_window_module.QMessageBox, "critical", lambda *args: None)

    _open(window, monkeypatch, invalid_path)

    assert _visible_state(window) == before

