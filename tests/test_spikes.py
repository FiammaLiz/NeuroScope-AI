import numpy as np
import pytest
from neuroscope.analysis.spikes import estimate_noise_mad, detect_negative_spikes


def test_mad_noise_estimator_is_positive():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 2, 10000)
    assert 1.8 < estimate_noise_mad(x) < 2.2


def test_negative_spike_detector_finds_inserted_events():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 5000)
    x[[1000, 3000]] = -10
    result = detect_negative_spikes(x, sampling_rate_hz=1000, threshold_sd=-4)
    assert 1000 in result["samples"]
    assert 3000 in result["samples"]


@pytest.mark.parametrize("signal,fs,threshold", [
    (np.array([1., 2.]), 0, -4), (np.array([1., np.nan]), 1000, -4),
    (np.array([1., 2.]), 1000, 0), (np.array([1., 2.]), np.inf, -4),
])
def test_detector_rejects_invalid_inputs(signal, fs, threshold):
    with pytest.raises(ValueError):
        detect_negative_spikes(signal, fs, threshold)


def test_detector_requires_explicit_channel_for_2d():
    data = np.array([[0., -10, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="explicit channel"):
        detect_negative_spikes(data, 1000)
    result = detect_negative_spikes(data, 1000, channel=0, channel_axis=0)
    assert result["samples"].tolist() == [1]
