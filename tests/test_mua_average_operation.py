from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from neuroscope.analysis.contracts import OperationContext, SampleInterval
from neuroscope.analysis.mua_average_operation import (
    AVERAGE_MUA_OPERATION,
    IMPLEMENTATION_VERSION,
    OPERATION_ID,
    OPERATION_VERSION,
    MUAArtifactReference,
    MUAAverageInput,
    MUAAverageParameters,
    register_mua_average,
)
from neuroscope.analysis.operations import (
    ApprovalRequiredError,
    ExecutionStatus,
    OperationRunner,
)
from neuroscope.analysis.registry import OperationRegistry
from neuroscope.exploration import MetadataStatus
from neuroscope.mua_artifacts import MUAArtifactStore


def _source(
    reference,
    channel,
    values,
    *,
    fs=1000.0,
    mode="rel",
    stdmin=-2.0,
    bandwidth=0.004,
    time_offset=0.0,
):
    params = {
        "stdmin": stdmin,
        "mua_window_s": bandwidth,
        "mode": mode,
        "num_points": 17,
    }
    metadata = {
        "method": "MATLAB calcmua_verFi.m",
        "operation_id": "neuroscope.mua.calc-mua",
        "operation_version": "1.0.0",
        "implementation_version": "calcMUA-matlab-mapping-v1",
        "source_sha256": "source-digest",
        "execution_id": f"execution-{channel}",
        "dataset_id": "dataset-A",
        "source_channel": channel,
        "sample_interval": {"start": 100, "stop": 100 + len(values)},
        "sampling_rate_hz": fs,
        "sampling_rate_status": "confirmed",
        "input_signal_kind": "raw_voltage",
        "input_signal_kind_status": "confirmed",
        "input_amplitude_units": "mV",
        "input_amplitude_units_status": "confirmed",
        "filtering_description": "bandpass 300-3000 Hz",
        "filtering_status": "confirmed",
        "parameters": params,
        "mua_units": "s^-1" if mode == "rel" else "events s^-1",
        "threshold": -0.5,
        "noise_estimate": 0.25,
    }
    times = (np.arange(1, len(values) + 1, dtype=np.float64) / fs) + time_offset
    artifact_reference = reference.write(
        mua=np.asarray(values, dtype=np.float64),
        times_so=times,
        spike_sample_indices=np.asarray([], dtype=np.int64),
        spike_times_s=np.asarray([], dtype=np.float64),
        spike_amplitudes=np.asarray([], dtype=np.float64),
        manifest=metadata,
    )
    model = MUAArtifactReference(
        artifact_reference=artifact_reference,
        execution_id=f"execution-{channel}",
        dataset_id="dataset-A",
        channel_index=channel,
        sample_interval=SampleInterval(start=100, stop=100 + len(values)),
        output_length=len(values),
        sampling_rate_hz=fs,
        sampling_rate_status=MetadataStatus.CONFIRMED,
        mua_units=metadata["mua_units"],
        amplitude_units="mV",
        amplitude_units_status=MetadataStatus.CONFIRMED,
        mode=mode,
        stdmin=stdmin,
        actual_threshold=-0.5,
        noise_estimate=0.25,
        bandwidth_s=bandwidth,
        num_points=17,
        source_sha256="source-digest",
        source_operation_id="neuroscope.mua.calc-mua",
        source_operation_version="1.0.0",
        source_implementation_version="calcMUA-matlab-mapping-v1",
        signal_kind="raw_voltage",
        signal_kind_status=MetadataStatus.CONFIRMED,
        filtering_description="bandpass 300-3000 Hz",
        filtering_status=MetadataStatus.CONFIRMED,
    )
    return model, np.asarray(values, dtype=np.float64), times


def _run(store: MUAArtifactStore, traces):
    registry = OperationRegistry()
    register_mua_average(registry)
    runner = OperationRunner(registry, mua_artifact_store=store)
    prepared = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        MUAAverageInput(traces=tuple(traces)),
        MUAAverageParameters(block_samples=2),
        context=OperationContext(),
    )
    with pytest.raises(ApprovalRequiredError):
        runner.execute(prepared)
    approved = runner.approve(prepared, approved_by="Researcher")
    record = runner.execute(approved)
    return record


def test_mean(tmp_path: Path):
    store = MUAArtifactStore(tmp_path / "a")
    a = _source(store, 0, [1, 2, 3])[0]
    b = _source(store, 1, [3, 4, 5])[0]
    before_a = store.open(a.artifact_reference)
    try:
        original = before_a.mua.copy()
    finally:
        before_a.close()
    registry = OperationRegistry()
    assert register_mua_average(registry) is AVERAGE_MUA_OPERATION
    assert registry.get(OPERATION_ID, version=OPERATION_VERSION).implementation_version == IMPLEMENTATION_VERSION
    with pytest.raises(ValueError, match="already registered"):
        register_mua_average(registry)

    first = _run(store, (a, b))
    second = _run(store, (b, a))
    assert first.status is second.status is ExecutionStatus.SUCCEEDED
    assert first.output.artifact_reference != a.artifact_reference
    with store.open(first.output.artifact_reference) as average:
        np.testing.assert_array_equal(average.mua, [2, 3, 4])
        expected = [
            item.execution_id
            for item in sorted((a, b), key=lambda item: item.artifact_reference)
        ]
        assert average.manifest["source_execution_ids"] == expected
        assert average.manifest["contributing_channels"] == 2
        assert average.manifest["weighting"] == "equal"
    with store.open(second.output.artifact_reference) as average:
        np.testing.assert_array_equal(average.mua, [2, 3, 4])
    after_a = store.open(a.artifact_reference)
    try:
        np.testing.assert_array_equal(after_a.mua, original)
    finally:
        after_a.close()
    snapshot = first.to_snapshot()
    assert "mua" not in snapshot["output"]
    assert first.output.approval_id == first.provenance.approval_id
    assert first.output.approved_by == "Researcher"
    assert first.output.source_thresholds == (-0.5, -0.5)
    assert first.output.source_noise_estimates == (0.25, 0.25)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sampling_rate_hz", 2000.0, "sampling_rate_hz"),
        ("mode", "abs", "mode"),
        ("stdmin", -3.0, "stdmin"),
        ("bandwidth_s", 0.01, "bandwidth_s"),
        ("dataset_id", "other-dataset", "dataset_id"),
        ("filtering_description", "highpass only", "filtering_description"),
    ],
)
def test_incompatible_inputs_are_rejected_before_preparation(tmp_path, field, value, message):
    store = MUAArtifactStore(tmp_path / "a")
    first, second = _source(store, 0, [1, 2, 3])[0], _source(store, 1, [1, 2, 3])[0]
    second = second.model_copy(update={field: value})
    with pytest.raises((ValidationError, ValueError), match=message):
        MUAAverageInput(traces=(first, second))


def test_nonfinite_input_trace_fails_without_dropping_samples(tmp_path):
    store = MUAArtifactStore(tmp_path / "a")
    first = _source(store, 0, [1.0, 2.0, 3.0])[0]
    second = _source(store, 1, [1.0, np.nan, 3.0])[0]
    record = _run(store, (first, second))
    assert record.status is ExecutionStatus.FAILED
    assert "nonfinite" in record.error.message.lower()
    assert len(list(store.root.glob("*/manifest.json"))) == 2


def test_actual_time_grid_mismatch_fails_after_approval(tmp_path):
    store = MUAArtifactStore(tmp_path / "a")
    first = _source(store, 0, [1.0, 2.0, 3.0])[0]
    second = _source(store, 1, [2.0, 3.0, 4.0], time_offset=0.0001)[0]
    record = _run(store, (first, second))
    assert record.status is ExecutionStatus.FAILED
    assert "temporal coordinate arrays" in record.error.message


def test_unknown_sampling_rate_is_not_accepted_for_time_grid_averaging(tmp_path):
    store = MUAArtifactStore(tmp_path / "a")
    first, second = _source(store, 0, [1, 2])[0], _source(store, 1, [2, 3])[0]
    first = first.model_copy(update={"sampling_rate_status": MetadataStatus.INFERRED})
    with pytest.raises(ValidationError, match="confirmed sampling rates"):
        MUAAverageInput(traces=(first, second))


def test_unknown_metadata_average(tmp_path):
    store = MUAArtifactStore(tmp_path / "a")
    first, second = _source(store, 0, [1, 2])[0], _source(store, 1, [2, 3])[0]
    first = first.model_copy(update={"filtering_status": MetadataStatus.UNKNOWN})
    with pytest.raises(ValidationError, match="filtering/preprocessing"):
        MUAAverageInput(traces=(first, second))
    first = _source(store, 0, [1, 2])[0].model_copy(
        update={"amplitude_units_status": MetadataStatus.UNKNOWN, "amplitude_units": None}
    )
    with pytest.raises(ValidationError, match="amplitude units"):
        MUAAverageInput(traces=(first, second))
