import numpy as np
import pytest
from pydantic import ValidationError

import neuroscope.analysis.signed_peak_operation as signed_module
from neuroscope.analysis.contracts import (
    DataReference,
    DatasetReference,
    MetadataSnapshot,
    OperationContext,
    SampleInterval,
)
from neuroscope.analysis.operations import OperationRunner
from neuroscope.analysis.registry import OperationRegistry
from neuroscope.analysis.signed_peak_operation import (
    FILTERING_WARNING,
    IMPLEMENTATION_VERSION,
    MAXIMUM_AMPLITUDE_FACTOR,
    NOISE_ESTIMATOR_DIVISOR,
    OPERATION_ID,
    OPERATION_VERSION,
    PROMINENCE_FACTOR,
    SOURCE_SHA256,
    SignedPeakInput,
    SignedPeakParameters,
    estimate_signed_peak_noise,
    register_signed_peak_detector,
)
from neuroscope.data_access import DatasetStore
from neuroscope.domain import DatasetMetadata, DatasetProfile
from neuroscope.exploration import DatasetExploration, MetadataStatus


def _setup(
    signal,
    *,
    filtering_description=None,
    filtering_status=MetadataStatus.UNKNOWN,
    sampling_rate=1000.0,
    sampling_rate_status=MetadataStatus.CONFIRMED,
    units="mV",
    units_status=MetadataStatus.CONFIRMED,
):
    values = np.asarray(signal, dtype=float)
    interval = SampleInterval(start=0, stop=len(values))
    confirmed = ["signal_kind"]
    inferred = []
    unknown = []
    for name, value, status in (
        ("sampling_rate_hz", sampling_rate, sampling_rate_status),
        ("units", units, units_status),
    ):
        if status is MetadataStatus.CONFIRMED:
            confirmed.append(name)
        elif status is MetadataStatus.INFERRED:
            inferred.append(name)
        else:
            unknown.append(name)
    if filtering_status is MetadataStatus.CONFIRMED:
        confirmed.append("filtering_description")
    elif filtering_status is MetadataStatus.INFERRED:
        inferred.append("filtering_description")
    else:
        unknown.append("filtering_description")
    profile = DatasetProfile(
        dataset_id="signed-test-data",
        name="signed detector test data",
        source_path="synthetic://signed-test-data",
        file_format="synthetic",
        shape=list(values.shape),
        dtype=str(values.dtype),
        metadata=DatasetMetadata(
            sampling_rate_hz=sampling_rate,
            units=units,
            signal_kind="raw_voltage",
            filtering_description=filtering_description,
        ),
        confirmed_fields=confirmed,
        inferred_fields=inferred,
        unknown_fields=unknown,
    )
    exploration = DatasetExploration(values, profile)
    dataset = DatasetReference(
        dataset_id=profile.dataset_id,
        source_reference="synthetic://signed-test-data",
    )
    store = DatasetStore()
    registration = store.register(exploration, dataset=dataset)
    registry = OperationRegistry()
    register_signed_peak_detector(registry)
    runner = OperationRunner(registry, data_store=store)
    metadata = tuple(
        MetadataSnapshot(
            name=name,
            value=exploration.summary.metadata[name].value,
            status=exploration.summary.metadata[name].status,
        )
        for name in (
            "signal_kind",
            "sampling_rate_hz",
            "units",
            "filtering_description",
        )
    )
    context = OperationContext(
        dataset_reference=dataset,
        metadata=metadata,
        selected_channels=(0,),
        sample_interval=interval,
    )
    reference = registration.reference((0,), interval)
    return values, runner, reference, context


def _execute(runner, reference, context, parameters):
    prepared = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        SignedPeakInput(signal=reference),
        parameters,
        context=context,
    )
    approved = runner.approve(prepared, approved_by="signed-detector-test")
    return prepared, runner.execute(approved)


def _peaks(signal, threshold, noise=1.0):
    values = np.asarray(signal, dtype=float)
    output = signed_module._detect_signed_peaks(values, threshold, noise)
    return output


def test_owner_noise_estimator_is_uncentered_median_absolute_amplitude():
    signal = np.asarray([-1.0, 0.0, 1.0, 2.0])
    assert estimate_signed_peak_noise(signal) == pytest.approx(
        np.median(np.abs(signal)) / NOISE_ESTIMATOR_DIVISOR
    )
    offset = signal + 10.0
    assert estimate_signed_peak_noise(offset) == pytest.approx(
        np.median(np.abs(offset)) / NOISE_ESTIMATOR_DIVISOR
    )
    assert estimate_signed_peak_noise(offset) != pytest.approx(
        estimate_signed_peak_noise(signal)
    )


def test_noise_estimate_changes_when_large_events_change_the_absolute_amplitude_median():
    sparse = np.asarray([0.0] * 8 + [10.0] * 2)
    dense = np.asarray([0.0] * 4 + [10.0] * 6)
    assert estimate_signed_peak_noise(dense) > estimate_signed_peak_noise(sparse)


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(-2.0, (5,)), (2.0, (15,))],
)
def test_signed_absolute_threshold_detects_each_requested_polarity(threshold, expected):
    signal = np.zeros(24)
    signal[5] = -3.0
    signal[15] = 3.0
    _, runner, reference, context = _setup(signal)
    params = SignedPeakParameters(
        threshold_mode="absolute", threshold=threshold, noise_override=1.0
    )

    prepared, record = _execute(runner, reference, context, params)

    assert prepared.operation_id == OPERATION_ID
    assert record.output.original_sample_indices == expected
    assert record.output.window_relative_sample_indices == expected
    assert record.output.polarity == ("negative" if threshold < 0 else "positive")
    assert record.output.peak_properties_signal == (
        "inverted" if threshold < 0 else "original"
    )


def test_height_prominence_properties_and_ordering_match_scipy_semantics():
    signal = np.zeros(32)
    signal[6] = 2.0
    signal[18] = 4.0
    peaks, amplitudes, properties, _ = _peaks(signal, 2.0)

    assert peaks.tolist() == [6, 18]
    assert amplitudes.tolist() == [2.0, 4.0]
    assert properties["peak_heights"].tolist() == [2.0, 4.0]
    assert properties["prominences"].tolist() == [2.0, 4.0]
    assert PROMINENCE_FACTOR * abs(2.0) == 1.5
    assert MAXIMUM_AMPLITUDE_FACTOR == 25.0

    # Height alone passes, but prominence below 0.75 * abs(threshold) rejects it.
    shallow = np.full(12, 2.4)
    shallow[5] = 2.9
    found, _, _, _ = _peaks(shallow, 2.0)
    assert found.size == 0


def test_amplitude_cap_is_strict_and_applied_after_peak_finding():
    signal = np.zeros(20)
    signal[5] = -25.0
    signal[12] = -24.999
    peaks, amplitudes, properties, _ = _peaks(signal, -2.0, noise=1.0)

    assert peaks.tolist() == [12]
    assert amplitudes.tolist() == [-24.999]
    assert properties["peak_heights"].tolist() == [24.999]


def test_scipy_endpoint_exclusion_and_plateau_selection_are_preserved():
    endpoints = np.zeros(14)
    endpoints[0] = -4.0
    endpoints[6] = -4.0
    endpoints[-1] = -5.0
    peaks, _, _, _ = _peaks(endpoints, -2.0, noise=1.0)
    assert peaks.tolist() == [6]

    plateau = np.zeros(14)
    plateau[6:8] = -4.0
    plateau_peaks, _, _, _ = _peaks(plateau, -2.0, noise=1.0)
    assert plateau_peaks.tolist() == [6]


def test_overlapping_synthetic_waveforms_can_collapse_to_one_scipy_peak():
    from neuroscope.data.synthetic import (
        SpikeTemplate,
        SyntheticRecordingConfig,
        SyntheticSpike,
        generate_synthetic_recording,
    )

    config = SyntheticRecordingConfig(
        channel_count=1,
        duration_s=0.04,
        sampling_rate_hz=1000.0,
        voltage_units="mV",
        seed=3,
        noise_std_by_channel=(0.1,),
        templates=(SpikeTemplate("overlap", (0.25, 1.0, 0.25), 1),),
        spikes=(
            SyntheticSpike("a", 0, 0.015, -2.0, "overlap"),
            SyntheticSpike("b", 0, 0.016, -2.0, "overlap"),
        ),
    )
    generated = generate_synthetic_recording(config)
    assert len(generated.ground_truth) == 2
    peaks, _, _, _ = _peaks(generated.recording.signal[0], -1.5, noise=1.0)
    assert len(peaks) == 1


def test_absolute_noise_override_changes_only_cap_not_threshold():
    signal = np.zeros(20)
    signal[8] = -10.0
    _, runner, reference, context = _setup(signal)

    kept = _execute(
        runner,
        reference,
        context,
        SignedPeakParameters(
            threshold_mode="absolute", threshold=-2.0, noise_override=1.0
        ),
    )[1].output
    removed = _execute(
        runner,
        reference,
        context,
        SignedPeakParameters(
            threshold_mode="absolute", threshold=-2.0, noise_override=0.1
        ),
    )[1].output

    assert kept.threshold == removed.threshold == -2.0
    assert kept.noise_source == removed.noise_source == "overridden"
    assert kept.noise_estimate == 1.0
    assert removed.noise_estimate == 0.1
    assert kept.window_relative_sample_indices == (8,)
    assert removed.window_relative_sample_indices == ()


def test_estimated_noise_and_multiplier_mode_preserve_source_formula():
    signal = np.tile(np.asarray([-1.0, 1.0]), 20)
    signal[20] = -8.0
    _, runner, reference, context = _setup(signal)
    expected_noise = np.median(np.abs(signal)) / NOISE_ESTIMATOR_DIVISOR
    params = SignedPeakParameters(threshold_mode="noise_multiplier", stdmin=-2.0)

    _, record = _execute(runner, reference, context, params)

    assert record.output.noise_source == "estimated"
    assert record.output.noise_estimate == pytest.approx(expected_noise)
    assert record.output.threshold == pytest.approx(-2.0 * expected_noise)
    assert record.output.polarity == "negative"
    assert 20 in record.output.window_relative_sample_indices


def test_absolute_threshold_mode_estimates_noise_when_override_is_omitted():
    signal = np.tile(np.asarray([-1.0, 1.0]), 20)
    signal[20] = -8.0
    _, runner, reference, context = _setup(signal)
    expected_noise = np.median(np.abs(signal)) / NOISE_ESTIMATOR_DIVISOR

    _, record = _execute(
        runner,
        reference,
        context,
        SignedPeakParameters(threshold_mode="absolute", threshold=-2.0),
    )

    assert record.output.noise_source == "estimated"
    assert record.output.noise_estimate == pytest.approx(expected_noise)
    assert record.output.threshold == -2.0
    assert 20 in record.output.window_relative_sample_indices


def test_zero_estimated_noise_with_absolute_threshold_is_removed_by_zero_cap():
    signal = np.zeros(16)
    signal[8] = -5.0
    _, runner, reference, context = _setup(signal)

    _, record = _execute(
        runner,
        reference,
        context,
        SignedPeakParameters(threshold_mode="absolute", threshold=-1.0),
    )

    assert record.output.noise_estimate == 0.0
    assert record.output.noise_source == "estimated"
    assert record.output.window_relative_sample_indices == ()


def test_zero_threshold_and_zero_multiplier_return_no_detections():
    signal = np.zeros(16)
    signal[8] = -5.0
    _, runner, reference, context = _setup(signal)
    absolute = _execute(
        runner,
        reference,
        context,
        SignedPeakParameters(threshold_mode="absolute", threshold=0.0),
    )[1].output
    multiplied = _execute(
        runner,
        reference,
        context,
        SignedPeakParameters(threshold_mode="noise_multiplier", stdmin=0.0),
    )[1].output

    assert absolute.window_relative_sample_indices == ()
    assert absolute.peak_properties.peak_heights is None
    assert multiplied.window_relative_sample_indices == ()
    assert multiplied.threshold == 0.0


def test_sample_times_are_unavailable_without_confirmed_rate_and_units_stay_unknown():
    signal = np.zeros(12)
    signal[5] = -4.0
    _, runner, reference, context = _setup(
        signal,
        sampling_rate=None,
        sampling_rate_status=MetadataStatus.UNKNOWN,
        units=None,
        units_status=MetadataStatus.UNKNOWN,
    )

    _, record = _execute(
        runner,
        reference,
        context,
        SignedPeakParameters(
            threshold_mode="absolute", threshold=-1.0, noise_override=1.0
        ),
    )

    assert record.output.times_s_relative_to_window is None
    assert record.output.sampling_rate_hz is None
    assert record.output.sampling_rate_status is MetadataStatus.UNKNOWN
    assert record.output.amplitude_units is None
    assert record.output.amplitude_units_status is MetadataStatus.UNKNOWN


def test_multiplier_magnitude_25_cannot_pass_height_and_strict_cap_together():
    signal = np.tile(np.asarray([-1.0, 1.0]), 20)
    signal[20] = -100.0
    noise = estimate_signed_peak_noise(signal)
    threshold = -25.0 * noise
    peaks, _, _, _ = _peaks(signal, threshold, noise=noise)
    assert peaks.size == 0


def test_endpoints_are_excluded_and_window_properties_remain_window_relative():
    signal = np.zeros(140)
    signal[100] = -5.0
    signal[101] = -1.0
    signal[139] = -6.0
    # Analyze [99, 120): the peak is local index 1, with a property base in this window.
    _, runner, reference, context = _setup(signal)
    subwindow = reference.model_copy(
        update={"sample_interval": SampleInterval(start=99, stop=120)}
    )
    context = context.model_copy(
        update={"sample_interval": SampleInterval(start=99, stop=120)}
    )
    # Isolate an interior event while retaining a true endpoint peak in the window.
    signal[100] = -5.0
    signal[101] = 0.0
    signal[139] = -6.0
    prepared, record = _execute(
        runner,
        subwindow,
        context,
        SignedPeakParameters(
            threshold_mode="absolute", threshold=-2.0, noise_override=1.0
        ),
    )

    assert prepared.warnings == (FILTERING_WARNING,)
    assert record.output.window_relative_sample_indices == (1,)
    assert record.output.original_sample_indices == (100,)
    assert record.output.peak_property_index_reference == "window_relative_samples"
    assert record.output.peak_properties.left_bases is not None
    assert all(0 <= index < 21 for index in record.output.peak_properties.left_bases)
    assert record.output.times_s_relative_to_window == pytest.approx((0.001,))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_signal_is_a_typed_execution_failure(bad):
    signal = np.zeros(12)
    signal[5] = bad
    _, runner, reference, context = _setup(signal)
    prepared = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        SignedPeakInput(signal=reference),
        SignedPeakParameters(threshold_mode="absolute", threshold=-1.0),
        context=context,
    )
    approval = runner.approve(prepared, approved_by="input-validation-test")
    record = runner.execute(approval)
    assert record.status.value == "failed"
    assert "finite real numeric" in record.error.message


def test_invalid_parameters_and_mode_mixing_are_rejected_before_prepare():
    with pytest.raises(ValidationError):
        SignedPeakParameters(threshold_mode="absolute", threshold=float("inf"))
    with pytest.raises(ValidationError):
        SignedPeakParameters(threshold_mode="absolute", threshold=1.0, noise_override=-1.0)
    with pytest.raises(ValidationError):
        SignedPeakParameters(
            threshold_mode="noise_multiplier", stdmin=-4.0, noise_override=1.0
        )
    with pytest.raises(ValidationError):
        SignedPeakParameters(threshold_mode="absolute", threshold=1.0, stdmin=-4.0)


def test_input_contract_rejects_more_than_one_selected_channel():
    _, _, reference, _ = _setup(np.zeros(8))
    multichannel_reference = DataReference(
        dataset=reference.dataset,
        session_handle=reference.session_handle,
        channels=(0, 1),
        sample_interval=reference.sample_interval,
    )
    with pytest.raises(ValidationError, match="exactly one channel"):
        SignedPeakInput(signal=multichannel_reference)


@pytest.mark.parametrize(
    ("description", "status", "warns"),
    [
        (None, MetadataStatus.UNKNOWN, True),
        ("Band-pass filter described, not confirmed", MetadataStatus.INFERRED, True),
        ("No filtering applied in this synthetic setup", MetadataStatus.CONFIRMED, False),
    ],
)
def test_filtering_metadata_warns_without_blocking_and_is_preserved(description, status, warns):
    signal = np.zeros(16)
    signal[7] = -4.0
    _, runner, reference, context = _setup(
        signal, filtering_description=description, filtering_status=status
    )
    prepared = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        SignedPeakInput(signal=reference),
        SignedPeakParameters(threshold_mode="absolute", threshold=-1.0, noise_override=1.0),
        context=context,
    )
    assert (FILTERING_WARNING in prepared.warnings) is warns
    record = runner.execute(runner.approve(prepared, approved_by="filter-metadata-test"))

    assert record.status.value == "succeeded"
    assert (FILTERING_WARNING in record.warnings) is warns
    assert record.output.filtering_description == description
    assert record.output.filtering_status is status
    assert record.output.algorithm_provenance.filtering_warning is warns
    assert any(item.name == "filtering_description" for item in record.output_metadata)


def test_missing_filtering_snapshot_warns_but_uses_registered_metadata():
    signal = np.zeros(16)
    signal[7] = -4.0
    _, runner, reference, context = _setup(signal)
    context = context.model_copy(
        update={
            "metadata": tuple(
                item for item in context.metadata if item.name != "filtering_description"
            )
        }
    )
    prepared = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        SignedPeakInput(signal=reference),
        SignedPeakParameters(
            threshold_mode="absolute", threshold=-1.0, noise_override=1.0
        ),
        context=context,
    )
    record = runner.execute(runner.approve(prepared, approved_by="missing-filter-test"))

    assert FILTERING_WARNING in prepared.warnings
    assert record.status.value == "succeeded"
    assert FILTERING_WARNING in record.warnings
    assert record.output.algorithm_provenance.filtering_warning is True
    assert record.output.filtering_status is MetadataStatus.UNKNOWN


def test_preparation_and_approval_do_not_run_peak_finder_and_approval_is_bound(monkeypatch):
    signal = np.zeros(20)
    signal[10] = -5.0
    _, runner, reference, context = _setup(signal)
    calls = []
    original = signed_module.find_peaks

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(signed_module, "find_peaks", counted)
    params = SignedPeakParameters(threshold_mode="absolute", threshold=-1.0, noise_override=1.0)
    prepared = runner.prepare(OPERATION_ID, OPERATION_VERSION, SignedPeakInput(signal=reference), params, context=context)
    assert calls == []
    approval = runner.approve(prepared, approved_by="approval-test")
    assert calls == []
    changed = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        SignedPeakInput(signal=reference),
        SignedPeakParameters(threshold_mode="absolute", threshold=-2.0, noise_override=1.0),
        context=context,
    )
    assert not approval.authorizes(changed)
    assert calls == []
    record = runner.execute(approval)
    assert record.status.value == "succeeded"
    assert len(calls) == 1


def test_original_input_is_unchanged_and_source_provenance_is_recorded():
    signal = np.zeros(24)
    signal[12] = 4.0
    original = signal.copy()
    _, runner, reference, context = _setup(signal)
    _, record = _execute(
        runner,
        reference,
        context,
        SignedPeakParameters(threshold_mode="absolute", threshold=1.0, noise_override=1.0),
    )

    np.testing.assert_array_equal(signal, original)
    assert record.operation_id == "neuroscope.spike.detect-signed-peaks"
    assert record.operation_version == "1.0.0"
    assert record.implementation_version == IMPLEMENTATION_VERSION
    assert record.output.algorithm_provenance.source_sha256 == SOURCE_SHA256
    assert record.output.algorithm_provenance.noise_estimator_divisor == 0.6745
    assert record.output.algorithm_provenance.prominence_factor == 0.75
    assert record.output.algorithm_provenance.maximum_amplitude_factor == 25.0
    assert record.input_data.signal == reference
    assert record.started_at_utc.tzinfo is not None
    assert record.finished_at_utc.tzinfo is not None


def test_registers_once_and_rejects_duplicate_operation_registration():
    registry = OperationRegistry()
    operation = register_signed_peak_detector(registry)
    assert registry.get(OPERATION_ID, version=OPERATION_VERSION) is operation
    with pytest.raises(ValueError, match="already registered"):
        register_signed_peak_detector(registry)
