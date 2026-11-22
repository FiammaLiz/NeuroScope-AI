import numpy as np
import pytest

from neuroscope.exploration import DatasetExploration
from neuroscope.navigation import NavigationState


def test_initial_state_shows_full_recording_and_all_original_channels():
    state = NavigationState(sample_count=20, channel_count=3)

    assert state.selected_channels == (0, 1, 2)
    assert (state.visible_window.start, state.visible_window.stop) == (0, 20)


def test_channel_changes_preserve_visible_interval_and_original_indices():
    state = NavigationState(sample_count=20, channel_count=4)
    state.zoom(5, 12)

    state.select_channels((3, 1))

    assert state.selected_channels == (3, 1)
    assert (state.visible_window.start, state.visible_window.stop) == (5, 12)


def test_empty_and_all_channel_selections_are_explicit():
    state = NavigationState(sample_count=10, channel_count=2)
    state.select_channels(())
    assert state.selected_channels == ()

    state.select_channels((0, 1))
    assert state.selected_channels == (0, 1)


@pytest.mark.parametrize("channels", [(-1,), (2,), (0, 0), (True,), (0.5,)])
def test_invalid_channel_selections_are_rejected(channels):
    state = NavigationState(sample_count=10, channel_count=2)

    with pytest.raises((TypeError, ValueError, IndexError)):
        state.select_channels(channels)


def test_zoom_pan_boundaries_and_reset():
    state = NavigationState(sample_count=100, channel_count=1)
    state.zoom(20, 40)
    state.pan(10)
    assert (state.visible_window.start, state.visible_window.stop) == (30, 50)
    state.pan(1000)
    assert (state.visible_window.start, state.visible_window.stop) == (80, 100)
    state.pan(-1000)
    assert (state.visible_window.start, state.visible_window.stop) == (0, 20)
    state.reset()
    assert (state.visible_window.start, state.visible_window.stop) == (0, 100)


@pytest.mark.parametrize(
    "start, stop",
    [(-1, 3), (0, 0), (5, 4), (0, 11)],
)
def test_empty_or_out_of_range_windows_are_rejected(start, stop):
    state = NavigationState(sample_count=10, channel_count=1)

    with pytest.raises((TypeError, ValueError)):
        state.zoom(start, stop)


def test_plot_coordinates_map_to_sample_windows_with_or_without_confirmed_rate():
    state = NavigationState(sample_count=100, channel_count=1)
    state.set_from_axis_limits(0.25, 0.75, sampling_rate_hz=4.0)
    assert (state.visible_window.start, state.visible_window.stop) == (1, 3)

    state.set_from_axis_limits(8.2, 12.1, sampling_rate_hz=None)
    assert (state.visible_window.start, state.visible_window.stop) == (8, 13)


def test_axis_limits_clamp_to_recording_and_reject_invalid_values():
    state = NavigationState(sample_count=10, channel_count=1)
    state.set_from_axis_limits(-20.0, 2.0, sampling_rate_hz=None)
    assert (state.visible_window.start, state.visible_window.stop) == (0, 2)
    state.set_from_axis_limits(9.5, 20.0, sampling_rate_hz=None)
    assert (state.visible_window.start, state.visible_window.stop) == (9, 10)
    with pytest.raises(ValueError):
        state.set_from_axis_limits(float("nan"), 2.0, sampling_rate_hz=None)
    with pytest.raises(ValueError):
        state.set_from_axis_limits(3.0, 2.0, sampling_rate_hz=None)


def test_navigation_views_and_repeated_window_changes_do_not_modify_source():
    original = np.arange(30, dtype=np.float64).reshape(3, 10)
    before = original.copy()
    exploration = DatasetExploration.from_array(original, channel_axis=0)
    state = NavigationState.from_exploration(exploration)

    for _ in range(4):
        state.select_channels((2, 0))
        state.zoom(2, 8)
        for channel in state.selected_channels:
            view = exploration.window(state.visible_window, channel_index=channel)
            assert not view.flags.writeable
        state.pan(1)
        state.reset()

    np.testing.assert_array_equal(original, before)

