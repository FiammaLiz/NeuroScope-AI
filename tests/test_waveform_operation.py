from uuid import uuid4

import h5py
import numpy as np
import pytest

from neuroscope.analysis.contracts import (
    DataReference,
    DatasetReference,
    MetadataSnapshot,
    OperationContext,
    SampleInterval,
)
from neuroscope.analysis.operations import (
    ApprovalRequiredError,
    ExecutionStatus,
    OperationRunner,
)
from neuroscope.analysis.registry import OperationRegistry
from neuroscope.analysis.signed_peak_operation import (
    OPERATION_VERSION as SIGNED_VERSION,
)
from neuroscope.analysis.signed_peak_operation import (
    SignedPeakInput,
    SignedPeakOutput,
    register_signed_peak_detector,
)
from neuroscope.analysis.spike_operation import (
    OPERATION_ID as NEGATIVE_ID,
)
from neuroscope.analysis.spike_operation import (
    OPERATION_VERSION as NEGATIVE_VERSION,
)
from neuroscope.analysis.spike_operation import (
    NegativeSpikeInput,
    NegativeSpikeOutput,
    register_negative_spike_detector,
)
from neuroscope.analysis.waveform_operation import (
    OPERATION_ID,
    OPERATION_VERSION,
    DetectedSpikeEvent,
    DetectionEventCollection,
    WaveformExtractionInput,
    WaveformExtractionParameters,
    detection_collection_from_record,
    register_waveform_extraction,
)
from neuroscope.artifacts import WaveformArtifactStore
from neuroscope.data.hdf5_source import open_matlab_v73
from neuroscope.data.synthetic import (
    SpikeTemplate,
    SyntheticRecordingConfig,
    SyntheticSpike,
    generate_synthetic_recording,
)
from neuroscope.data_access import DatasetStore
from neuroscope.domain import DatasetMetadata, DatasetProfile
from neuroscope.exploration import DatasetExploration, MetadataStatus


def _setup(values, tmp_path, *, rate=None, units=None, lazy_source=None):
    metadata = DatasetMetadata(
        sampling_rate_hz=rate,
        units=units,
        signal_kind="raw_voltage",
    )
    confirmed = tuple(
        name
        for name, value in (("sampling_rate_hz", rate), ("units", units))
        if value is not None
    ) + ("signal_kind",)
    unknown = tuple(
        name for name, value in (("sampling_rate_hz", rate), ("units", units))
        if value is None
    )
    exploration = DatasetExploration.from_array(
        lazy_source if lazy_source is not None else values,
        channel_axis=0 if len(values.shape) == 2 else None,
        metadata=metadata,
        confirmed_fields=confirmed,
        unknown_fields=unknown,
    )
    store = DatasetStore()
    registration = store.register(exploration)
    reference = registration.reference(
        tuple(range(registration.channel_count)),
        SampleInterval(start=0, stop=registration.sample_count),
    )
    # Keep the artifact root short for Windows' per-process path-length limit.
    artifact_store = WaveformArtifactStore(tmp_path.parent / "w")
    registry = OperationRegistry()
    register_waveform_extraction(registry)
    runner = OperationRunner(registry, data_store=store, artifact_store=artifact_store)
    context = OperationContext(
        dataset_reference=registration.dataset,
        selected_channels=reference.channels,
        sample_interval=reference.sample_interval,
        metadata=(
            MetadataSnapshot(
                name="sampling_rate_hz", value=rate, status=(
                    MetadataStatus.CONFIRMED if rate is not None else MetadataStatus.UNKNOWN
                )
            ),
            MetadataSnapshot(
                name="units", value=units, status=(
                    MetadataStatus.CONFIRMED if units is not None else MetadataStatus.UNKNOWN
                )
            ),
        ),
    )
    return store, registration, reference, artifact_store, runner, context


def _detections(reference, events, *, rate=None, units=None, rate_status=None, units_status=None):
    return DetectionEventCollection(
        dataset=reference.dataset,
        source_data_reference=reference,
        detector_operation_id="neuroscope.spike.detect-negative",
        detector_operation_version="1.0.0",
        detection_reference="execution://detector-run",
        sampling_rate_hz=rate,
        sampling_rate_status=rate_status or (
            MetadataStatus.CONFIRMED if rate is not None else MetadataStatus.UNKNOWN
        ),
        amplitude_units=units,
        amplitude_units_status=units_status or (
            MetadataStatus.CONFIRMED if units is not None else MetadataStatus.UNKNOWN
        ),
        events=tuple(events),
    )


def _run(reference, detections, runner, context, *, pre=2, post=2):
    prepared = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        WaveformExtractionInput(signal=reference, detections=detections),
        WaveformExtractionParameters(pre_samples=pre, post_samples=post),
        context=context,
    )
    approval = runner.approve(prepared, approved_by="researcher")
    return runner.execute(approval)


def test_extracts_original_samples_polarities_channels_and_population_summaries(tmp_path):
    values = np.asarray(
        [
            [0, 1, 5, -4, 2, 7, 1, 0, 0],
            [0, -3, 2, 8, 1, -2, 5, 0, 0],
        ],
        dtype=np.float32,
    )
    original = values.copy()
    store, registration, reference, artifacts, runner, context = _setup(
        values, tmp_path, rate=1000.0, units="uV"
    )
    detections = _detections(
        reference,
        [
            DetectedSpikeEvent(event_id="neg", channel_index=0, original_sample_index=3,
                               window_relative_sample_index=3, polarity="negative"),
            DetectedSpikeEvent(event_id="pos", channel_index=1, original_sample_index=3,
                               window_relative_sample_index=3, polarity="positive"),
            DetectedSpikeEvent(event_id="neg2", channel_index=0, original_sample_index=4,
                               window_relative_sample_index=4, polarity="negative"),
        ],
        rate=1000.0,
        units="uV",
    )

    record = _run(reference, detections, runner, context)

    assert record.status is ExecutionStatus.SUCCEEDED, record.error
    output = record.output
    assert output.waveform_shape == (3, 5)
    assert output.event_sample_offset == 2
    with artifacts.open(output.artifact_reference) as artifact:
        np.testing.assert_array_equal(
            artifact.waveforms,
            [[1, 5, -4, 2, 7], [-3, 2, 8, 1, -2], [5, -4, 2, 7, 1]],
        )
        manifest_events = artifact.manifest["accepted_events"]
        assert [item["event_id"] for item in manifest_events] == ["neg", "pos", "neg2"]
        assert [item["original_sample_index"] for item in manifest_events] == [3, 3, 4]
        assert [item["channel_index"] for item in manifest_events] == [0, 1, 0]
        assert manifest_events[0]["sample_time_seconds_relative_to_recording_sample_zero"] == 0.003
        assert artifact.manifest["sampling_rate_status"] == "confirmed"
        assert artifact.manifest["amplitude_units"] == "uV"
        np.testing.assert_array_equal(
            artifact.measurement("minimum_position_from_event"), [0, -2, -1]
        )
        np.testing.assert_array_equal(
            artifact.measurement("maximum_position_from_event"), [2, 0, 1]
        )
        channel_zero = artifact.channel_summary(0)
        np.testing.assert_allclose(channel_zero["mean"], [3, 0.5, -1, 4.5, 4])
        np.testing.assert_allclose(channel_zero["standard-deviation"], [2, 4.5, 3, 2.5, 3])
        np.testing.assert_allclose(channel_zero["median"], [3, 0.5, -1, 4.5, 4])
        assert artifact.manifest["channel_summaries"]["0"]
    np.testing.assert_array_equal(values, original)
    assert "waveforms.npy" not in record.output.model_dump_json()
    assert "artifact://waveforms/" in record.output.artifact_reference
    assert registration.sample_count == 9
    store.unregister(registration.session_handle)


def test_boundaries_are_excluded_without_truncation_or_padding(tmp_path):
    values = np.arange(8, dtype=np.float64).reshape(1, 8)
    store, registration, reference, artifacts, runner, context = _setup(values, tmp_path)
    detections = _detections(
        reference,
        [
            DetectedSpikeEvent(event_id="left", channel_index=0, original_sample_index=0),
            DetectedSpikeEvent(event_id="inside", channel_index=0, original_sample_index=3),
            DetectedSpikeEvent(event_id="right", channel_index=0, original_sample_index=7),
        ],
    )
    record = _run(reference, detections, runner, context)
    assert record.status is ExecutionStatus.SUCCEEDED
    assert record.output.accepted_event_count == 1
    assert record.output.excluded_event_count == 2
    with artifacts.open(record.output.artifact_reference) as artifact:
        assert [entry["event_id"] for entry in artifact.manifest["excluded_events"]] == [
            "left", "right"
        ]
        assert [entry["reason"] for entry in artifact.manifest["excluded_events"]] == [
            "pre_window_outside_recording", "post_window_outside_recording"
        ]
        np.testing.assert_array_equal(artifact.waveforms, [[1, 2, 3, 4, 5]])
    store.unregister(registration.session_handle)


def test_empty_detection_result_has_valid_empty_artifact_and_unknown_metadata(tmp_path):
    values = np.zeros((1, 6), dtype=np.float32)
    store, registration, reference, artifacts, runner, context = _setup(values, tmp_path)
    record = _run(reference, _detections(reference, []), runner, context)
    assert record.status is ExecutionStatus.SUCCEEDED
    assert record.output.waveform_shape == (0, 5)
    assert record.output.sampling_rate_hz is None
    assert record.output.sampling_rate_status is MetadataStatus.UNKNOWN
    assert record.output.amplitude_units is None
    with artifacts.open(record.output.artifact_reference) as artifact:
        assert artifact.waveforms.shape == (0, 5)
        assert artifact.manifest["accepted_events"] == []
    store.unregister(registration.session_handle)


def test_waveform_extraction_preserves_synthetic_ground_truth_sample_alignment(tmp_path):
    config = SyntheticRecordingConfig(
        channel_count=1,
        duration_s=2.0,
        sampling_rate_hz=10.0,
        voltage_units="uV",
        seed=7,
        noise_std_by_channel=(0.0,),
        templates=(SpikeTemplate("simple", (0.25, 1.0, 0.5), peak_index=1),),
        spikes=(SyntheticSpike("truth-1", 0, 0.6, -2.0, "simple"),),
    )
    synthetic = generate_synthetic_recording(config)
    truth = synthetic.ground_truth[0]
    assert truth.sample_index == 6
    store, registration, reference, artifacts, runner, context = _setup(
        synthetic.recording.signal,
        tmp_path,
        rate=config.sampling_rate_hz,
        units=config.voltage_units,
    )
    detections = _detections(
        reference,
        [DetectedSpikeEvent(event_id="detected-1", channel_index=0,
                             original_sample_index=truth.sample_index,
                             polarity=truth.polarity)],
        rate=config.sampling_rate_hz,
        units=config.voltage_units,
    )
    record = _run(reference, detections, runner, context, pre=1, post=1)
    assert record.status is ExecutionStatus.SUCCEEDED, record.error
    with artifacts.open(record.output.artifact_reference) as artifact:
        np.testing.assert_array_equal(artifact.waveforms, [[-0.5, -2.0, -1.0]])
        assert artifact.manifest["accepted_events"][0]["original_sample_index"] == 6
        assert artifact.manifest["accepted_events"][0]["detector_time_seconds_relative_to_window"] is None
        assert artifact.manifest["accepted_events"][0][
            "sample_time_seconds_relative_to_recording_sample_zero"
        ] == truth.timestamp_s
    store.unregister(registration.session_handle)


def test_preparation_and_approval_do_not_read_samples_and_unapproved_execution_fails(
    tmp_path, monkeypatch
):
    values = np.arange(20, dtype=np.float32).reshape(1, 20)
    store, registration, reference, _artifacts, runner, context = _setup(values, tmp_path)
    detections = _detections(
        reference,
        [DetectedSpikeEvent(event_id="event", channel_index=0, original_sample_index=10)],
    )
    reads = []
    original_window = DatasetExploration.window

    def track_window(self, window, **kwargs):
        reads.append((window.start, window.stop, kwargs.get("channel_index")))
        return original_window(self, window, **kwargs)

    monkeypatch.setattr(DatasetExploration, "window", track_window)
    prepared = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        WaveformExtractionInput(signal=reference, detections=detections),
        WaveformExtractionParameters(pre_samples=2, post_samples=2),
        context=context,
    )
    assert reads == []
    with pytest.raises(ApprovalRequiredError):
        runner.execute(prepared)
    approval = runner.approve(prepared, approved_by="researcher")
    assert reads == []
    record = runner.execute(approval)
    assert record.status is ExecutionStatus.SUCCEEDED
    assert reads == [(8, 13, 0)]
    store.unregister(registration.session_handle)


def test_stale_registration_fails_execution_and_does_not_resolve_replacement(tmp_path):
    values = np.arange(12, dtype=np.float32).reshape(1, 12)
    store, registration, reference, _artifacts, runner, context = _setup(values, tmp_path)
    detections = _detections(
        reference,
        [DetectedSpikeEvent(event_id="event", channel_index=0, original_sample_index=5)],
    )
    prepared = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        WaveformExtractionInput(signal=reference, detections=detections),
        WaveformExtractionParameters(pre_samples=1, post_samples=1),
        context=context,
    )
    approval = runner.approve(prepared, approved_by="researcher")
    store.unregister(registration.session_handle)
    failed = runner.execute(approval)
    assert failed.status is ExecutionStatus.FAILED
    assert failed.error.error_type == "DatasetAccessError"


def test_invalid_event_sample_and_wrong_dataset_are_rejected(tmp_path):
    values = np.arange(10, dtype=np.float32).reshape(1, 10)
    store, registration, reference, _artifacts, _runner, _context = _setup(values, tmp_path)
    with pytest.raises(ValueError, match="outside its source sample interval"):
        _detections(
            reference,
            [DetectedSpikeEvent(event_id="bad", channel_index=0, original_sample_index=10)],
        )
    other = DataReference(
        dataset=reference.dataset.model_copy(update={"dataset_id": "wrong"}),
        session_handle=reference.session_handle,
        channels=reference.channels,
        sample_interval=reference.sample_interval,
    )
    with pytest.raises(ValueError, match="same dataset"):
        WaveformExtractionInput(signal=other, detections=_detections(reference, []))
    store.unregister(registration.session_handle)


def test_source_detector_output_adapter_preserves_both_detector_event_models():
    """The collection adapter accepts the two distinct typed detector outputs."""
    assert NegativeSpikeOutput is not SignedPeakOutput
    # Full operation executions are covered by their own tests; schema field parity
    # prevents this handoff from depending on signal values or rerunning detection.
    common = {
        "channel_index", "original_sample_indices", "window_relative_sample_indices",
        "sampling_rate_hz", "sampling_rate_status", "amplitude_units",
        "amplitude_units_status", "times_s_relative_to_window",
    }
    assert common <= set(NegativeSpikeOutput.model_fields)
    assert common <= set(SignedPeakOutput.model_fields)
    assert "polarity" in SignedPeakOutput.model_fields


@pytest.mark.parametrize(
    ("operation_id", "parameters", "expected_polarity"),
    [
        ("neuroscope.spike.detect-negative", {"threshold_sd": -4.0}, "negative"),
        (
            "neuroscope.spike.detect-signed-peaks",
            {"threshold_mode": "absolute", "threshold": -0.5, "noise_override": 0.2},
            "negative",
        ),
    ],
)
def test_completed_detector_runs_adapt_to_waveform_event_collections(
    operation_id, parameters, expected_polarity
):
    values = np.tile(np.asarray([-0.2, 0.1, 0.2, -0.1], dtype=np.float64), 20)
    values[30] = -2.0
    profile = DatasetProfile(
        dataset_id="detector-handoff",
        name="detector handoff fixture",
        source_path="synthetic://detector-handoff",
        file_format="synthetic",
        shape=list(values.shape),
        dtype=str(values.dtype),
        metadata=DatasetMetadata(
            sampling_rate_hz=1000.0,
            units="mV",
            signal_kind="raw_voltage",
        ),
        confirmed_fields=["sampling_rate_hz", "units", "signal_kind"],
        unknown_fields=["filtering_description"],
    )
    exploration = DatasetExploration(values, profile)
    dataset = DatasetReference(
        dataset_id=profile.dataset_id,
        source_reference="synthetic://detector-handoff",
    )
    dataset_store = DatasetStore()
    registration = dataset_store.register(exploration, dataset=dataset)
    reference = registration.reference((0,), SampleInterval(start=0, stop=len(values)))
    registry = OperationRegistry()
    register_negative_spike_detector(registry)
    register_signed_peak_detector(registry)
    detector_runner = OperationRunner(registry, data_store=dataset_store)
    context = OperationContext(
        dataset_reference=dataset,
        selected_channels=(0,),
        sample_interval=reference.sample_interval,
        metadata=tuple(
            MetadataSnapshot(name=name, value=value.value, status=value.status)
            for name, value in exploration.summary.metadata.items()
            if name in {"signal_kind", "sampling_rate_hz", "units", "filtering_description"}
        ),
    )
    if operation_id == NEGATIVE_ID:
        model, version = NegativeSpikeInput(signal=reference), NEGATIVE_VERSION
    else:
        model, version = SignedPeakInput(signal=reference), SIGNED_VERSION
    prepared = detector_runner.prepare(
        operation_id,
        version,
        model,
        parameters,
        context=context,
    )
    execution = detector_runner.execute(
        detector_runner.approve(prepared, approved_by="researcher")
    )
    assert execution.status is ExecutionStatus.SUCCEEDED, execution.error
    collection = detection_collection_from_record(execution)
    assert collection.detector_operation_id == operation_id
    assert collection.events
    assert collection.events[0].channel_index == 0
    assert collection.events[0].original_sample_index == 30
    assert collection.events[0].polarity == expected_polarity
    dataset_store.unregister(registration.session_handle)


def test_lazy_hdf5_extraction_reads_only_waveform_windows(tmp_path, monkeypatch):
    path = tmp_path / "synthetic-large.mat"
    source_signal = np.arange(2_000_000, dtype=np.float32).reshape(2, -1)
    with h5py.File(path, "w", userblock_size=512) as h5:
        dataset = h5.create_dataset(
            "signal", data=source_signal.T, chunks=(4096, 2), compression="gzip"
        )
        dataset.attrs["MATLAB_class"] = np.bytes_("single")
    with path.open("r+b") as stream:
        stream.write(b"MATLAB 7.3 MAT-file".ljust(116, b" ") + b"\0" * 8 + b"\x00\x02IM")
    lazy = open_matlab_v73(path, "signal")
    store, registration, reference, artifacts, runner, context = _setup(
        source_signal, tmp_path, lazy_source=lazy
    )
    reads = []
    original_getitem = h5py.Dataset.__getitem__

    def track_read(dataset, selection):
        reads.append(selection)
        return original_getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", track_read)
    detections = _detections(
        reference,
        [
            DetectedSpikeEvent(event_id="a", channel_index=0, original_sample_index=100),
                DetectedSpikeEvent(event_id="b", channel_index=0, original_sample_index=750_000),
        ],
    )
    record = _run(reference, detections, runner, context, pre=2, post=3)
    assert record.status is ExecutionStatus.SUCCEEDED
    assert len(reads) == 2
    assert all(selection[0].stop - selection[0].start == 6 for selection in reads)
    assert sum(selection[0].stop - selection[0].start for selection in reads) == 12
    with artifacts.open(record.output.artifact_reference) as artifact:
        assert artifact.waveforms.shape == (2, 6)
    store.unregister(registration.session_handle)
    assert lazy.closed


def test_detection_collection_rejects_duplicate_ids():
    event = DetectedSpikeEvent(event_id="duplicate", channel_index=0, original_sample_index=1)
    with pytest.raises(ValueError, match="identifiers must be unique"):
        DetectionEventCollection(
            dataset=DatasetReference(dataset_id="data"),
            source_data_reference=DataReference(
                dataset=DatasetReference(dataset_id="data"),
                session_handle=uuid4(),
                channels=(0,),
                sample_interval=SampleInterval(start=0, stop=4),
            ),
            detector_operation_id="neuroscope.spike.detect-negative",
            detector_operation_version="1.0.0",
            detection_reference="execution://x",
            sampling_rate_status=MetadataStatus.UNKNOWN,
            amplitude_units_status=MetadataStatus.UNKNOWN,
            events=(event, event),
        )
