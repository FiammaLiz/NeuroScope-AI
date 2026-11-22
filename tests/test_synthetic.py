from dataclasses import replace
from math import inf, nan, nextafter
from pathlib import Path

import numpy as np
import pytest

from neuroscope.data.loaders import load_array
from neuroscope.data.synthetic import (
    SpikeTemplate,
    SyntheticRecordingConfig,
    SyntheticSpike,
    generate_synthetic_recording,
)


def make_config(*, seed: int = 12, noise: tuple[float, ...] = (0.0, 0.0)):
    template = SpikeTemplate("tri_phasic", (0.25, 1.0, 0.5), peak_index=1)
    spikes = (
        SyntheticSpike("s-positive", 0, 0.5, 2.0, "tri_phasic"),
        SyntheticSpike("s-negative", 1, 1.0, -3.0, "tri_phasic"),
    )
    return SyntheticRecordingConfig(
        channel_count=2,
        duration_s=2.0,
        sampling_rate_hz=10.0,
        voltage_units="uV",
        seed=seed,
        noise_std_by_channel=noise,
        templates=(template,),
        spikes=spikes,
    )


def test_generator_is_deterministic_and_truth_is_separate():
    first = generate_synthetic_recording(make_config())
    second = generate_synthetic_recording(make_config())
    np.testing.assert_array_equal(first.recording.signal, second.recording.signal)
    assert first.ground_truth == second.ground_truth
    assert first.recording.config == second.recording.config
    assert first.recording.signal.shape == (2, 20)
    assert len(first.ground_truth) == 2


def test_noise_seed_and_per_channel_noise():
    config_a = make_config(seed=1, noise=(1.0, 2.0))
    config_b = make_config(seed=2, noise=(1.0, 2.0))
    result_a = generate_synthetic_recording(config_a)
    result_b = generate_synthetic_recording(config_b)
    assert not np.array_equal(result_a.recording.signal, result_b.recording.signal)
    assert not np.array_equal(result_a.recording.signal[0], result_a.recording.signal[1])
    assert result_a.ground_truth == result_b.ground_truth


def test_overlapping_spikes_add_waveforms_on_their_selected_channel():
    config = make_config(
        noise=(0.0, 0.0),
    )
    config = replace(
        config,
        spikes=(
            SyntheticSpike("overlap-a", 0, 0.5, 2.0, "tri_phasic"),
            SyntheticSpike("overlap-b", 0, 0.6, 4.0, "tri_phasic"),
        ),
    )

    result = generate_synthetic_recording(config)

    np.testing.assert_array_equal(result.recording.signal[0, 4:8], [0.5, 3.0, 5.0, 2.0])
    np.testing.assert_array_equal(result.recording.signal[1], np.zeros(20))


def test_adding_spike_does_not_change_noise_on_another_channel():
    noise_only = replace(make_config(seed=31, noise=(1.0, 1.0)), spikes=())
    with_spike = replace(
        noise_only,
        spikes=(SyntheticSpike("channel-zero", 0, 0.5, 2.0, "tri_phasic"),),
    )

    baseline = generate_synthetic_recording(noise_only).recording.signal
    changed = generate_synthetic_recording(with_spike).recording.signal

    np.testing.assert_array_equal(changed[1], baseline[1])


def test_signal_metadata_and_signed_spike_ground_truth():
    result = generate_synthetic_recording(make_config())
    assert np.isfinite(result.recording.signal).all()
    assert result.recording.metadata.sampling_rate_hz == 10.0
    assert result.recording.metadata.units == "uV"
    assert result.recording.metadata.channel_axis == 0
    assert result.recording.metadata.time_axis == 1
    positive, negative = result.ground_truth
    assert (positive.spike_id, positive.channel_index, positive.sample_index) == (
        "s-positive", 0, 5
    )
    assert positive.timestamp_s == positive.sample_index / 10.0
    assert (positive.amplitude, positive.polarity, positive.template_id) == (
        2.0, "positive", "tri_phasic"
    )
    assert (negative.spike_id, negative.channel_index, negative.sample_index) == (
        "s-negative", 1, 10
    )
    assert (negative.amplitude, negative.polarity) == (-3.0, "negative")
    np.testing.assert_array_equal(result.recording.signal[0, 4:7], [0.5, 2.0, 1.0])
    np.testing.assert_array_equal(result.recording.signal[1, 9:12], [-0.75, -3.0, -1.5])


def test_timestamp_is_quantized_to_nearest_sample():
    config = make_config()
    spike = SyntheticSpike("half-sample", 0, 0.55, 1.0, "tri_phasic")
    config = SyntheticRecordingConfig(
        channel_count=config.channel_count,
        duration_s=config.duration_s,
        sampling_rate_hz=config.sampling_rate_hz,
        voltage_units=config.voltage_units,
        seed=config.seed,
        noise_std_by_channel=config.noise_std_by_channel,
        templates=config.templates,
        spikes=(spike,),
    )
    truth = generate_synthetic_recording(config).ground_truth[0]
    assert truth.sample_index == 6  # ties round upward
    assert truth.timestamp_s == 0.6


def test_half_sample_neighbors_follow_round_half_up():
    config = make_config()
    half_sample_time = 0.55

    assert config.timestamp_to_sample(nextafter(half_sample_time, 0.0)) == 5
    assert config.timestamp_to_sample(half_sample_time) == 6
    assert config.timestamp_to_sample(nextafter(half_sample_time, inf)) == 6


def test_timestamps_near_recording_end_do_not_truncate_templates():
    config = make_config()
    valid_near_end = SyntheticSpike("valid-end", 0, 1.8, 1.0, "tri_phasic")
    valid_config = replace(config, spikes=(valid_near_end,))
    result = generate_synthetic_recording(valid_config)
    assert result.ground_truth[0].sample_index == 18
    assert result.ground_truth[0].timestamp_s == 1.8

    rounds_outside = SyntheticSpike(
        "outside-end", 0, nextafter(config.duration_s, 0.0), 1.0, "tri_phasic"
    )
    with pytest.raises(ValueError, match="boundaries"):
        replace(config, spikes=(rounds_outside,))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"channel_count": 0}, "channel_count"),
        ({"duration_s": 0.0}, "duration_s"),
        ({"sampling_rate_hz": 0.0}, "sampling_rate_hz"),
        ({"voltage_units": " "}, "voltage_units"),
        ({"seed": -1}, "seed"),
        ({"noise_std_by_channel": (1.0,)}, "one value per channel"),
        ({"noise_std_by_channel": (-1.0, 0.0)}, "non-negative"),
        ({"duration_s": 2.05}, "integer sample count"),
    ],
)
def test_invalid_recording_configuration_is_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(make_config(), **changes)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"duration_s": nan}, "duration_s must be finite"),
        ({"sampling_rate_hz": inf}, "sampling_rate_hz must be finite"),
        ({"noise_std_by_channel": (nan, 0.0)}, "noise standard deviation must be finite"),
    ],
)
def test_non_finite_recording_configuration_is_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(make_config(), **changes)


@pytest.mark.parametrize(
    "timestamp_s, amplitude",
    [(nan, 1.0), (0.5, inf)],
)
def test_non_finite_spike_values_are_rejected(timestamp_s, amplitude):
    with pytest.raises(ValueError, match="must be finite"):
        SyntheticSpike("non-finite", 0, timestamp_s, amplitude, "tri_phasic")


def test_invalid_templates_and_spikes_are_rejected():
    with pytest.raises(ValueError, match="unit positive maximum"):
        SpikeTemplate("bad", (0.2, 0.8, 0.1), peak_index=1)
    with pytest.raises(ValueError, match="outside the recording"):
        replace(
            make_config(),
            spikes=(SyntheticSpike("late", 0, 2.0, 1.0, "tri_phasic"),),
        )
    with pytest.raises(ValueError, match="boundaries"):
        replace(
            make_config(),
            spikes=(SyntheticSpike("edge", 0, 0.0, 1.0, "tri_phasic"),),
        )
    with pytest.raises(ValueError, match="unknown template"):
        replace(
            make_config(),
            spikes=(SyntheticSpike("unknown", 0, 1.0, 1.0, "missing"),),
        )


def test_duplicate_event_ids_are_rejected():
    duplicate_ids = (
        SyntheticSpike("duplicate", 0, 0.5, 1.0, "tri_phasic"),
        SyntheticSpike("duplicate", 1, 1.0, 1.0, "tri_phasic"),
    )
    with pytest.raises(ValueError, match="spike_id values must be unique"):
        replace(make_config(), spikes=duplicate_ids)


def test_export_round_trips_through_normal_loader_without_truth(tmp_path: Path):
    dataset = generate_synthetic_recording(make_config())
    output = tmp_path / "synthetic.npy"
    dataset.recording.save_npy(output)
    loaded = load_array(output)
    np.testing.assert_array_equal(loaded, dataset.recording.signal)
    assert not hasattr(loaded, "ground_truth")

    original_bytes = output.read_bytes()
    with pytest.raises(FileExistsError):
        dataset.recording.save_npy(output)
    assert output.read_bytes() == original_bytes

    unrelated = tmp_path / "unrelated.npy"
    unrelated_bytes = b"unrelated existing file contents"
    unrelated.write_bytes(unrelated_bytes)
    with pytest.raises(FileExistsError):
        dataset.recording.save_npy(unrelated)
    assert unrelated.read_bytes() == unrelated_bytes

    replacement = generate_synthetic_recording(make_config(seed=99))
    replacement.recording.save_npy(output, overwrite=True)
    np.testing.assert_array_equal(load_array(output), replacement.recording.signal)

    wrong_extension = tmp_path / "synthetic.csv"
    with pytest.raises(ValueError, match=".npy extension"):
        dataset.recording.save_npy(wrong_extension)
    assert not wrong_extension.exists()
    with pytest.raises(TypeError, match="overwrite must be a bool"):
        dataset.recording.save_npy(tmp_path / "bad-flag.npy", overwrite="yes")
