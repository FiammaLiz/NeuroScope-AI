"""Numerical tests for the mapped MATLAB calcMUA calculation."""

import numpy as np
import pytest

from neuroscope.analysis.mua import calc_mua


def _background(count: int = 32) -> np.ndarray:
    return np.tile(np.asarray([0.1, -0.1]), count // 2).astype(np.float64)


def test_positive_and_negative_peak_conventions_and_kde_units():
    positive = _background()
    positive[10] = 2.0
    positive_result = calc_mua(
        positive,
        sampling_rate_hz=1000.0,
        stdmin=2.0,
        mua_window_s=0.002,
        mode="rel",
        num_points=7,
    )
    noise = 0.1 / 0.6745
    assert positive_result.noise_estimates[0] == pytest.approx(noise)
    assert positive_result.thresholds[0] == pytest.approx(2 * noise)
    assert positive_result.spike_sample_indices[0].tolist() == [10]
    assert positive_result.spike_times_s[0].tolist() == pytest.approx([0.011])
    assert positive_result.spike_amplitudes[0].tolist() == [2.0]
    assert positive_result.evaluation_times_s.tolist() == pytest.approx(
        np.arange(1, 33) / 1000
    )
    assert positive_result.mua.shape == (32, 1)
    expected_peak_density = 1 / (0.002 * np.sqrt(2 * np.pi))
    assert positive_result.mua[10, 0] == pytest.approx(expected_peak_density)

    absolute = calc_mua(
        positive,
        sampling_rate_hz=1000.0,
        stdmin=2.0,
        mua_window_s=0.002,
        mode="abs",
        num_points=7,
    )
    assert np.array_equal(absolute.mua, positive_result.mua)
    assert absolute.mode == "abs"

    negative = _background()
    negative[11] = -2.0
    negative_result = calc_mua(
        negative,
        sampling_rate_hz=1000.0,
        stdmin=-2.0,
        mua_window_s=0.002,
        mode="rel",
        num_points=1,
    )
    assert negative_result.spike_sample_indices[0].tolist() == [11]
    assert negative_result.spike_amplitudes[0].tolist() == [-2.0]
    assert negative_result.spike_times_s[0].tolist() == pytest.approx([0.012])


def test_abs_mode_scales_density_by_detected_event_count_and_num_points_is_redundant():
    values = _background(64)
    values[[10, 30]] = [2.0, 2.5]
    relative = calc_mua(
        values,
        sampling_rate_hz=2000.0,
        stdmin=2.0,
        mua_window_s=0.001,
        mode="rel",
        num_points=4,
    )
    absolute = calc_mua(
        values,
        sampling_rate_hz=2000.0,
        stdmin=2.0,
        mua_window_s=0.001,
        mode="abs",
        num_points=800,
    )
    assert relative.spike_sample_indices[0].tolist() == [10, 30]
    assert np.allclose(absolute.mua[:, 0], relative.mua[:, 0] * 2)
    assert relative.mua.shape == (64, 1)
    assert absolute.mua.shape == (64, 1)


def test_fractional_sample_bandwidth_is_used_in_seconds_without_rounding():
    values = _background()
    values[10] = 2.0
    bandwidth_s = 0.0015  # 1.5 samples at this explicitly supplied sampling rate.
    result = calc_mua(
        values,
        sampling_rate_hz=1000.0,
        stdmin=2.0,
        mua_window_s=bandwidth_s,
        mode="rel",
        num_points=32,
    )
    assert result.mua[10, 0] == pytest.approx(
        1 / (bandwidth_s * np.sqrt(2 * np.pi))
    )


def test_kde_sums_overlapping_event_kernels_without_realigning_events():
    values = _background()
    values[[10, 12]] = [2.0, 2.5]
    result = calc_mua(
        values,
        sampling_rate_hz=1000.0,
        stdmin=2.0,
        mua_window_s=0.005,
        mode="abs",
        num_points=32,
    )
    assert result.spike_sample_indices[0].tolist() == [10, 12]
    assert result.spike_times_s[0].tolist() == pytest.approx([0.011, 0.013])
    expected_at_sample_10 = 1 + np.exp(-0.5 * ((0.011 - 0.013) / 0.005) ** 2)
    expected_at_sample_10 /= 0.005 * np.sqrt(2 * np.pi)
    assert result.mua[10, 0] == pytest.approx(expected_at_sample_10)


def test_no_events_return_nan_trace_and_matlab_nan_times():
    result = calc_mua(
        np.zeros(8),
        sampling_rate_hz=25000.0,
        stdmin=-4.0,
        mua_window_s=0.01,
        mode="rel",
        num_points=9,
    )
    assert result.spike_sample_indices[0].size == 0
    assert np.isnan(result.mua[:, 0]).all()
    assert np.isnan(result.times_so[:, 0]).all()
    assert result.evaluation_times_s[0] == pytest.approx(1 / 25000)


def test_boundary_peaks_are_excluded_and_plateau_uses_first_sample():
    values = _background()
    values[0] = 2.0
    values[-1] = 3.0
    values[12:15] = 2.5
    result = calc_mua(
        values,
        sampling_rate_hz=1000.0,
        stdmin=2.0,
        mua_window_s=0.005,
        mode="rel",
        num_points=32,
    )
    assert result.spike_sample_indices[0].tolist() == [12]
    assert result.spike_amplitudes[0].tolist() == [2.5]


def test_strict_amplitude_cap_and_large_multiplier_constraint():
    noise = 0.1 / 0.6745
    cap = 25.0 * noise
    at_cap = _background()
    at_cap[10] = cap
    result = calc_mua(
        at_cap,
        sampling_rate_hz=1000.0,
        stdmin=2.0,
        mua_window_s=0.002,
        mode="rel",
        num_points=32,
    )
    assert result.spike_sample_indices[0].size == 0

    large_multiplier = calc_mua(
        at_cap,
        sampling_rate_hz=1000.0,
        stdmin=25.0,
        mua_window_s=0.002,
        mode="rel",
        num_points=32,
    )
    assert large_multiplier.spike_sample_indices[0].size == 0


def test_zero_multiplier_preserves_source_positive_peak_branch():
    values = np.full(32, -0.1)
    values[10] = 2.0
    values[20] = -2.0
    result = calc_mua(
        values,
        sampling_rate_hz=1000.0,
        stdmin=0.0,
        mua_window_s=0.002,
        mode="rel",
        num_points=32,
    )
    assert result.thresholds[0] == 0.0
    assert result.spike_sample_indices[0].tolist() == [10]


def test_channels_are_processed_independently_and_baseline_changes_noise():
    values = np.column_stack((_background(), -_background()))
    values[10, 0] = 2.0
    values[20, 1] = -2.0
    result = calc_mua(
        values,
        sampling_rate_hz=1000.0,
        stdmin=2.0,
        mua_window_s=0.002,
        mode="rel",
        num_points=32,
    )
    assert [indices.tolist() for indices in result.spike_sample_indices] == [[10], []]

    negative = _background()
    negative[20] = -2.0
    two_channels = np.column_stack((values[:, 0], negative))
    both = calc_mua(
        two_channels,
        sampling_rate_hz=1000.0,
        stdmin=-2.0,
        mua_window_s=0.002,
        mode="rel",
        num_points=32,
    )
    assert [indices.tolist() for indices in both.spike_sample_indices] == [[], [20]]

    shifted = _background() + 1.0
    shifted[10] = 3.0
    shifted_result = calc_mua(
        shifted,
        sampling_rate_hz=1000.0,
        stdmin=2.0,
        mua_window_s=0.002,
        mode="rel",
        num_points=32,
    )
    assert shifted_result.noise_estimates[0] != pytest.approx(0.1 / 0.6745)


@pytest.mark.parametrize(
    "signal,kwargs,error,match",
    [
        (np.empty(0), {}, ValueError, "empty"),
        (np.zeros((2, 3, 4)), {}, ValueError, "1D or 2D"),
        (np.asarray([1.0, np.nan]), {}, ValueError, "finite"),
        (np.asarray([1.0, np.inf]), {}, ValueError, "finite"),
        (np.asarray([1 + 2j]), {}, TypeError, "real numeric"),
        (np.ones(4), {"sampling_rate_hz": 0}, ValueError, "sampling_rate_hz"),
        (np.ones(4), {"mua_window_s": 0}, ValueError, "mua_window_s"),
        (np.ones(4), {"stdmin": np.inf}, ValueError, "stdmin"),
        (np.ones(4), {"num_points": 0}, ValueError, "num_points"),
        (np.ones(4), {"mode": "invalid"}, ValueError, "mode"),
    ],
)
def test_invalid_signal_and_parameters_are_rejected(signal, kwargs, error, match):
    parameters = {
        "sampling_rate_hz": 1000.0,
        "stdmin": -4.0,
        "mua_window_s": 0.005,
        "mode": "rel",
        "num_points": 50,
        **kwargs,
    }
    with pytest.raises(error, match=match):
        calc_mua(signal, **parameters)


def test_result_arrays_are_read_only_and_input_is_unchanged():
    values = _background()
    values[10] = 2.0
    original = values.copy()
    result = calc_mua(
        values,
        sampling_rate_hz=1000.0,
        stdmin=2.0,
        mua_window_s=0.002,
        mode="rel",
        num_points=32,
    )
    assert np.array_equal(values, original)
    with pytest.raises(ValueError):
        result.mua[0, 0] = 0.0
