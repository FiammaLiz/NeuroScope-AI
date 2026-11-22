from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from neuroscope.analysis.contracts import (
    MetadataSnapshot,
    OperationContext,
    SampleInterval,
)
from neuroscope.analysis.operations import (
    ApprovalRequiredError,
    OperationRunner,
    OperationValidationError,
)
from neuroscope.analysis.registry import OperationRegistry
from neuroscope.analysis.spike_operation import (
    IMPLEMENTATION_VERSION,
    NEGATIVE_SPIKE_DETECTOR,
    OPERATION_ID,
    OPERATION_VERSION,
    NegativeSpikeInput,
    NegativeSpikeOutput,
    NegativeSpikeParameters,
    register_negative_spike_detector,
)
from neuroscope.analysis.spikes import detect_negative_spikes
from neuroscope.data_access import DatasetStore
from neuroscope.domain import DatasetMetadata, DatasetProfile
from neuroscope.exploration import DatasetExploration, MetadataStatus


def _exploration(
    values: np.ndarray,
    *,
    channel_axis: int | None = None,
    sampling_rate_hz: float | None = None,
    sampling_rate_status: MetadataStatus = MetadataStatus.UNKNOWN,
    units: str | None = None,
    units_status: MetadataStatus = MetadataStatus.UNKNOWN,
    signal_kind: str | None = "raw_voltage",
    signal_kind_status: MetadataStatus = MetadataStatus.CONFIRMED,
) -> DatasetExploration:
    profile = DatasetProfile.from_array(
        Path("operation-input.npy"), values, channel_axis=channel_axis
    )
    metadata = DatasetMetadata(
        sampling_rate_hz=sampling_rate_hz,
        units=units,
        signal_kind=signal_kind,
        channel_axis=channel_axis,
        time_axis=1 - channel_axis if channel_axis is not None else None,
    )
    confirmed = []
    inferred = []
    unknown = []
    if values.ndim == 2:
        confirmed.append("channel_axis")
    for name, status in (
        ("sampling_rate_hz", sampling_rate_status),
        ("units", units_status),
        ("signal_kind", signal_kind_status),
    ):
        if status is MetadataStatus.CONFIRMED:
            confirmed.append(name)
        elif status is MetadataStatus.INFERRED:
            inferred.append(name)
        else:
            unknown.append(name)
    profile = profile.model_copy(
        update={
            "metadata": metadata,
            "confirmed_fields": confirmed,
            "inferred_fields": inferred,
            "unknown_fields": unknown,
        }
    )
    return DatasetExploration(values, profile)


def _metadata_context(exploration: DatasetExploration) -> OperationContext:
    fields = ("signal_kind", "sampling_rate_hz", "units")
    return OperationContext(
        metadata=tuple(
            MetadataSnapshot(
                name=name,
                value=exploration.summary.metadata[name].value,
                status=exploration.summary.metadata[name].status,
            )
            for name in fields
        )
    )


def _setup(
    values: np.ndarray,
    *,
    channel_axis: int | None = None,
    channel: int = 0,
    start: int = 0,
    stop: int | None = None,
    sampling_rate_hz: float | None = None,
    sampling_rate_status: MetadataStatus = MetadataStatus.UNKNOWN,
    units: str | None = None,
    units_status: MetadataStatus = MetadataStatus.UNKNOWN,
    signal_kind: str | None = "raw_voltage",
    signal_kind_status: MetadataStatus = MetadataStatus.CONFIRMED,
):
    exploration = _exploration(
        values,
        channel_axis=channel_axis,
        sampling_rate_hz=sampling_rate_hz,
        sampling_rate_status=sampling_rate_status,
        units=units,
        units_status=units_status,
        signal_kind=signal_kind,
        signal_kind_status=signal_kind_status,
    )
    store = DatasetStore()
    registration = store.register(exploration)
    sample_count = exploration.summary.sample_count
    reference = registration.reference(
        (channel,),
        SampleInterval(start=start, stop=sample_count if stop is None else stop),
    )
    registry = OperationRegistry()
    register_negative_spike_detector(registry)
    runner = OperationRunner(registry, data_store=store)
    context = _metadata_context(exploration)
    return exploration, store, registration, reference, runner, context


def _prepare(runner, reference, context, *, threshold_sd=-4.0):
    return runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        {"signal": reference},
        {"threshold_sd": threshold_sd},
        context=context,
    )


def test_production_operation_registers_once_with_stable_identity_and_contract():
    registry = OperationRegistry()
    definition = register_negative_spike_detector(registry)

    assert definition is NEGATIVE_SPIKE_DETECTOR
    assert registry.get(OPERATION_ID, version=OPERATION_VERSION) is definition
    assert definition.implementation_version == IMPLEMENTATION_VERSION
    assert definition.input_model is NegativeSpikeInput
    assert definition.parameters_model is NegativeSpikeParameters
    assert definition.output_model is NegativeSpikeOutput
    assert definition.parameters_model().threshold_sd == -4.0
    assert any("negative-going" in warning for warning in definition.warnings)
    assert any("refractory" in item for item in definition.limitations)
    with pytest.raises(ValueError, match="already registered"):
        register_negative_spike_detector(registry)


def test_adapter_matches_direct_detector_exactly_for_baseline_and_thresholds():
    rng = np.random.default_rng(48)
    values = 5.0 + rng.normal(0, 0.35, size=(2, 600))
    values[1, [100, 300, 500]] = [-8.0, -9.0, -7.0]
    original = values.copy()
    exploration, store, _, reference, runner, context = _setup(
        values,
        channel_axis=0,
        channel=1,
        sampling_rate_hz=20_000.0,
        sampling_rate_status=MetadataStatus.CONFIRMED,
        units="mV",
        units_status=MetadataStatus.CONFIRMED,
    )
    window = store.resolve(reference).channel(1).samples

    for multiplier in (-4.0, -5.5):
        direct = detect_negative_spikes(
            window,
            sampling_rate_hz=20_000.0,
            threshold_sd=multiplier,
        )
        record = runner.execute(
            runner.approve(
                _prepare(runner, reference, context, threshold_sd=multiplier),
                approved_by="researcher",
            )
        )
        output = record.output

        np.testing.assert_array_equal(
            output.window_relative_sample_indices, direct["samples"]
        )
        np.testing.assert_array_equal(output.original_sample_indices, direct["samples"])
        np.testing.assert_array_equal(
            output.times_s_relative_to_window, direct["times_s"]
        )
        np.testing.assert_array_equal(output.amplitudes, direct["amplitudes"])
        assert output.noise_sigma == direct["noise_sigma"]
        assert output.threshold == direct["threshold"]
        assert output.threshold_sd == direct["threshold_sd"] == multiplier
        assert output.channel_index == 1
        assert output.sample_interval == reference.sample_interval
        assert output.sampling_rate_hz == 20_000.0
        assert output.sampling_rate_status is MetadataStatus.CONFIRMED
        assert output.amplitude_units == "mV"
        assert output.amplitude_units_status is MetadataStatus.CONFIRMED
        assert record.data_references == (reference,)
        assert record.operation_id == OPERATION_ID
        assert record.operation_version == OPERATION_VERSION
        assert record.implementation_version == IMPLEMENTATION_VERSION
        assert record.parameters.threshold_sd == multiplier
        assert record.input_metadata == context.metadata
        assert record.started_at_utc.tzinfo is not None
        assert record.finished_at_utc.tzinfo is not None
        assert any("negative-going" in warning for warning in record.warnings)
        assert record.status.value == "succeeded"

    np.testing.assert_array_equal(values, original)
    assert exploration.dataset_id == reference.dataset.dataset_id


def test_subinterval_preserves_window_relative_and_original_indices():
    values = np.zeros(120, dtype=np.float64)
    values[53] = -10.0
    values[91] = -12.0
    _, _, _, reference, runner, context = _setup(
        values,
        start=40,
        stop=80,
        sampling_rate_hz=1000.0,
        sampling_rate_status=MetadataStatus.CONFIRMED,
    )
    record = runner.execute(
        runner.approve(_prepare(runner, reference, context), approved_by="researcher")
    )

    assert record.output.window_relative_sample_indices == (13,)
    assert record.output.original_sample_indices == (53,)
    assert record.output.times_s_relative_to_window == (0.013,)
    assert record.output.time_reference == "window_start_sample_zero"
    assert record.output.sample_interval == SampleInterval(start=40, stop=80)


def test_full_recording_reference_preserves_sample_zero_time_convention():
    values = np.zeros(50, dtype=np.float64)
    values[12] = -5.0
    _, _, _, reference, runner, context = _setup(
        values,
        sampling_rate_hz=2500.0,
        sampling_rate_status=MetadataStatus.CONFIRMED,
    )
    record = runner.execute(
        runner.approve(_prepare(runner, reference, context), approved_by="researcher")
    )

    assert reference.sample_interval == SampleInterval(start=0, stop=50)
    assert record.output.window_relative_sample_indices == (12,)
    assert record.output.original_sample_indices == (12,)
    assert record.output.times_s_relative_to_window == (12 / 2500.0,)


def test_unknown_rate_and_units_remain_unknown_and_times_are_unavailable():
    values = np.zeros(64, dtype=np.float64)
    values[20] = -4.0
    _, _, _, reference, runner, context = _setup(values)
    record = runner.execute(
        runner.approve(_prepare(runner, reference, context), approved_by="researcher")
    )

    assert record.output.window_relative_sample_indices == (20,)
    assert record.output.original_sample_indices == (20,)
    assert record.output.times_s_relative_to_window is None
    assert record.output.sampling_rate_hz is None
    assert record.output.sampling_rate_status is MetadataStatus.UNKNOWN
    assert record.output.amplitude_units is None
    assert record.output.amplitude_units_status is MetadataStatus.UNKNOWN


def test_inferred_rate_is_preserved_but_not_used_to_fabricate_seconds():
    values = np.zeros(32, dtype=np.float64)
    values[8] = -3.0
    _, _, _, reference, runner, context = _setup(
        values,
        sampling_rate_hz=1000.0,
        sampling_rate_status=MetadataStatus.INFERRED,
    )
    record = runner.execute(
        runner.approve(_prepare(runner, reference, context), approved_by="researcher")
    )

    assert record.output.sampling_rate_hz == 1000.0
    assert record.output.sampling_rate_status is MetadataStatus.INFERRED
    assert record.output.times_s_relative_to_window is None
    assert record.output.window_relative_sample_indices == (8,)


@pytest.mark.parametrize(
    "values, start, stop, expected",
    [
        (np.ones(2), 0, 2, ()),
        (np.zeros(16), 0, 16, ()),
        (np.array([0.0, -1.0, 0.0]), 0, 3, (1,)),
        (np.array([0.0, -1e-290, 0.0]), 0, 3, (1,)),
    ],
)
def test_short_empty_zero_and_near_zero_noise_match_direct_detector(
    values, start, stop, expected
):
    _, store, _, reference, runner, context = _setup(
        values, start=start, stop=stop
    )
    window = store.resolve(reference).channel(0).samples
    direct = detect_negative_spikes(
        window, sampling_rate_hz=None, threshold_sd=-4.0
    )
    record = runner.execute(
        runner.approve(_prepare(runner, reference, context), approved_by="researcher")
    )

    assert tuple(direct["samples"].tolist()) == expected
    assert record.output.window_relative_sample_indices == expected
    assert record.output.original_sample_indices == tuple(
        start + sample for sample in expected
    )
    assert record.output.times_s_relative_to_window is None
    assert record.output.noise_sigma == direct["noise_sigma"]
    assert record.output.threshold == direct["threshold"]


def test_first_and_last_samples_of_a_window_remain_excluded():
    values = np.array([-10.0, 0.0, 0.0, -12.0])
    _, store, _, reference, runner, context = _setup(values)
    direct = detect_negative_spikes(
        store.resolve(reference).channel(0).samples,
        sampling_rate_hz=None,
        threshold_sd=-4.0,
    )
    record = runner.execute(
        runner.approve(_prepare(runner, reference, context), approved_by="researcher")
    )

    assert direct["samples"].tolist() == []
    assert record.output.window_relative_sample_indices == ()
    assert record.output.original_sample_indices == ()


def test_approval_does_not_authorize_a_changed_window_reference():
    values = np.zeros(24, dtype=np.float64)
    _, _, registration, reference, runner, context = _setup(values, stop=20)
    prepared = _prepare(runner, reference, context)
    approval = runner.approve(prepared, approved_by="researcher")
    changed_reference = registration.reference(
        (0,), SampleInterval(start=1, stop=21)
    )
    changed_request = _prepare(runner, changed_reference, context)

    assert not approval.authorizes(changed_request)
    with pytest.raises(ApprovalRequiredError):
        runner.execute(changed_request)


def test_preparation_and_approval_do_not_call_detector_or_resolve_signal(monkeypatch):
    values = np.zeros(40, dtype=np.float64)
    values[20] = -8.0
    _, store, _, reference, runner, context = _setup(values)
    resolve_calls = []
    original_resolve = store.resolve

    def resolve_spy(selected_reference):
        resolve_calls.append(selected_reference)
        return original_resolve(selected_reference)

    monkeypatch.setattr(store, "resolve", resolve_spy)
    called = []

    def spy(*args, **kwargs):
        called.append(True)
        return detect_negative_spikes(*args, **kwargs)

    import neuroscope.analysis.spike_operation as adapter

    monkeypatch.setattr(adapter, "detect_negative_spikes", spy)
    prepared = _prepare(runner, reference, context)
    assert called == []
    assert resolve_calls == []
    approval = runner.approve(prepared, approved_by="researcher")
    assert called == []
    assert resolve_calls == []
    assert approval.prepared.input_data.signal == reference

    with pytest.raises(ApprovalRequiredError):
        runner.execute(prepared)
    assert called == []
    assert resolve_calls == []
    record = runner.execute(approval)
    assert called == [True]
    assert resolve_calls == [reference]
    assert record.status.value == "succeeded"


def test_invalid_channel_count_and_threshold_parameters_are_rejected():
    values = np.zeros((2, 10), dtype=np.float64)
    _, _, registration, _, runner, context = _setup(values, channel_axis=0)
    multiple = registration.reference(
        (0, 1), SampleInterval(start=0, stop=10)
    )
    with pytest.raises(OperationValidationError, match="exactly one channel"):
        _prepare(runner, multiple, context)

    for invalid in (0.0, 1.0, float("nan"), float("inf"), "-4", True):
        with pytest.raises((ValidationError, OperationValidationError, TypeError)):
            NegativeSpikeParameters(threshold_sd=invalid)


@pytest.mark.parametrize("invalid_selection", ["channel", "interval"])
def test_invalid_channel_or_interval_is_rejected_before_detector_execution(
    invalid_selection, monkeypatch
):
    values = np.zeros((2, 12), dtype=np.float64)
    _, store, registration, _, runner, context = _setup(values, channel_axis=0)
    if invalid_selection == "channel":
        reference = registration.reference(
            (2,), SampleInterval(start=0, stop=8)
        )
    else:
        reference = registration.reference(
            (0,), SampleInterval(start=4, stop=13)
        )
    called = []

    def spy(*args, **kwargs):
        called.append(True)
        return detect_negative_spikes(*args, **kwargs)

    import neuroscope.analysis.spike_operation as adapter

    monkeypatch.setattr(adapter, "detect_negative_spikes", spy)
    record = runner.execute(
        runner.approve(_prepare(runner, reference, context), approved_by="researcher")
    )

    assert record.status.value == "failed"
    assert record.error.error_type == "DatasetAccessError"
    assert called == []
    assert store is not None


def test_unknown_or_unsupported_signal_kind_is_not_silently_accepted():
    values = np.zeros(20, dtype=np.float64)
    _, _, _, reference, runner, context = _setup(
        values,
        signal_kind="spike_times",
        signal_kind_status=MetadataStatus.CONFIRMED,
    )
    prepared = _prepare(runner, reference, context)
    record = runner.execute(
        runner.approve(prepared, approved_by="researcher")
    )

    assert record.status.value == "failed"
    assert record.error.error_type == "ValueError"
    assert "raw_voltage" in record.error.message

    _, _, _, unknown_reference, unknown_runner, unknown_context = _setup(
        values,
        signal_kind=None,
        signal_kind_status=MetadataStatus.UNKNOWN,
    )
    with pytest.raises(OperationValidationError, match="signal_kind"):
        _prepare(unknown_runner, unknown_reference, unknown_context)


def test_invalidated_data_reference_prevents_detection_and_preserves_provenance():
    values = np.zeros(40, dtype=np.float64)
    values[15] = -10
    _, store, registration, reference, runner, context = _setup(values)
    prepared = _prepare(runner, reference, context)
    approved = runner.approve(prepared, approved_by="researcher")
    store.unregister(registration.session_handle)

    record = runner.execute(approved)

    assert record.status.value == "failed"
    assert record.error.error_type == "DatasetAccessError"
    assert record.data_references == (reference,)
    assert record.provenance.approval_id == approved.approval_id
    assert record.operation_id == OPERATION_ID


def test_operation_metadata_snapshot_must_match_registered_dataset():
    values = np.zeros(40, dtype=np.float64)
    _, _, _, reference, runner, context = _setup(
        values,
        sampling_rate_hz=1000.0,
        sampling_rate_status=MetadataStatus.CONFIRMED,
    )
    mismatched = OperationContext(
        metadata=tuple(
            item.model_copy(update={"value": 2000.0})
            if item.name == "sampling_rate_hz"
            else item
            for item in context.metadata
        )
    )
    record = runner.execute(
        runner.approve(_prepare(runner, reference, mismatched), approved_by="researcher")
    )

    assert record.status.value == "failed"
    assert "does not match" in record.error.message
