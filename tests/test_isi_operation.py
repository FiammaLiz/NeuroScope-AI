from uuid import uuid4

import numpy as np
import pytest

from neuroscope.analysis.contracts import (
    DataReference,
    DatasetReference,
    MetadataSnapshot,
    OperationContext,
    SampleInterval,
)
from neuroscope.analysis.isi_operation import (
    IMPLEMENTATION_VERSION,
    OPERATION_ID,
    OPERATION_VERSION,
    ISIInput,
    ISIParameters,
    isi_collection_from_detection_collection,
    isi_collection_from_record,
    register_isi_analysis,
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
    register_negative_spike_detector,
)
from neuroscope.analysis.waveform_operation import (
    DetectedSpikeEvent,
    DetectionEventCollection,
)
from neuroscope.artifacts import ISIArtifactStore
from neuroscope.data_access import DatasetStore
from neuroscope.domain import DatasetMetadata, DatasetProfile
from neuroscope.exploration import DatasetExploration, MetadataStatus


def _collection(
    events,
    *,
    sample_count=100,
    rate=1000.0,
    rate_status=MetadataStatus.CONFIRMED,
    detector_id="neuroscope.spike.detect-negative",
):
    dataset = DatasetReference(dataset_id="isi-dataset", source_reference="synthetic://isi")
    source = DataReference(
        dataset=dataset,
        session_handle=uuid4(),
        channels=(0, 1),
        sample_interval=SampleInterval(start=0, stop=sample_count),
    )
    waveform_handoff = DetectionEventCollection(
        dataset=dataset,
        source_data_reference=source,
        detector_operation_id=detector_id,
        detector_operation_version="1.0.0",
        detection_reference="execution://detector-123",
        sampling_rate_hz=rate,
        sampling_rate_status=rate_status,
        amplitude_units=None,
        amplitude_units_status=MetadataStatus.UNKNOWN,
        events=tuple(events),
    )
    return isi_collection_from_detection_collection(waveform_handoff)


def _setup(tmp_path):
    registry = OperationRegistry()
    definition = register_isi_analysis(registry)
    store = ISIArtifactStore(tmp_path.parent / "isi-artifacts")
    runner = OperationRunner(registry, isi_artifact_store=store)
    return definition, store, runner


def _context(collection):
    return OperationContext(
        dataset_reference=collection.dataset,
        selected_channels=collection.source_channels,
        sample_interval=collection.source_sample_interval,
        metadata=(
            MetadataSnapshot(
                name="sampling_rate_hz",
                value=collection.sampling_rate_hz,
                status=collection.sampling_rate_status,
            ),
        ),
    )


def _run(collection, params, runner):
    prepared = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        ISIInput(events=collection),
        params,
        context=_context(collection),
    )
    approval = runner.approve(prepared, approved_by="researcher")
    return runner.execute(approval), prepared


def _event(event_id, channel, sample, polarity="negative"):
    return DetectedSpikeEvent(
        event_id=event_id,
        channel_index=channel,
        original_sample_index=sample,
        polarity=polarity,
    )


def test_operation_registration_versions_and_detector_handoff_are_typed(tmp_path):
    registry = OperationRegistry()
    definition = register_isi_analysis(registry)
    assert registry.get(OPERATION_ID, version="1.0.0") is definition
    assert definition.version == OPERATION_VERSION == "1.0.0"
    assert definition.implementation_version == IMPLEMENTATION_VERSION
    for detector in (
        "neuroscope.spike.detect-negative",
        "neuroscope.spike.detect-signed-peaks",
    ):
        collection = _collection([_event("ev", 0, 10)], detector_id=detector)
        assert collection.detector_operation_id == detector
        result, _ = _run(collection, ISIParameters(polarity_mode="separate"),
                         OperationRunner(registry, isi_artifact_store=ISIArtifactStore(tmp_path.parent / f"{detector[-1]}")))
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output.detection_reference == "execution://detector-123"
        assert result.output.detector_execution_id == "detector-123"
        assert result.output.event_count == 1


@pytest.mark.parametrize(
    ("detector_id", "detector_version", "parameters"),
    [
        (NEGATIVE_ID, NEGATIVE_VERSION, {"threshold_sd": -4.0}),
        (
            "neuroscope.spike.detect-signed-peaks",
            SIGNED_VERSION,
            {"threshold_mode": "absolute", "threshold": -0.5, "noise_override": 0.2},
        ),
    ],
)
def test_isi_adapts_real_detector_execution_without_resolving_recording(
    tmp_path, detector_id, detector_version, parameters
):
    values = np.asarray(
        [(-0.2, 0.1, 0.2, -0.1)[index % 4] for index in range(80)],
        dtype=np.float64,
    )
    values[30] = -2.0
    profile = DatasetProfile(
        dataset_id="isi-detector-handoff",
        name="synthetic detector handoff",
        source_path="synthetic://isi-detector-handoff",
        file_format="synthetic",
        shape=[len(values)],
        dtype="float64",
        metadata=DatasetMetadata(
            sampling_rate_hz=1000.0, units="mV", signal_kind="raw_voltage"
        ),
        confirmed_fields=["sampling_rate_hz", "units", "signal_kind"],
        unknown_fields=["filtering_description"],
    )
    exploration = DatasetExploration(values, profile)
    data_store = DatasetStore()
    registration = data_store.register(
        exploration,
        dataset=DatasetReference(
            dataset_id=profile.dataset_id,
            source_reference="synthetic://isi-detector-handoff",
        ),
    )
    reference = registration.reference((0,), SampleInterval(start=0, stop=len(values)))
    registry = OperationRegistry()
    register_negative_spike_detector(registry)
    register_signed_peak_detector(registry)
    detector_runner = OperationRunner(registry, data_store=data_store)
    context = OperationContext(
        dataset_reference=registration.dataset,
        selected_channels=(0,),
        sample_interval=reference.sample_interval,
        metadata=tuple(
            MetadataSnapshot(name=name, value=value.value, status=value.status)
            for name, value in exploration.summary.metadata.items()
            if name in {"signal_kind", "sampling_rate_hz", "units", "filtering_description"}
        ),
    )
    detector_input = (
        NegativeSpikeInput(signal=reference)
        if detector_id == NEGATIVE_ID
        else SignedPeakInput(signal=reference)
    )
    detector_prepared = detector_runner.prepare(
        detector_id,
        detector_version,
        detector_input,
        parameters,
        context=context,
    )
    detector_record = detector_runner.execute(
        detector_runner.approve(detector_prepared, approved_by="researcher")
    )
    assert detector_record.status is ExecutionStatus.SUCCEEDED, detector_record.error
    isi_collection = isi_collection_from_record(detector_record)
    assert isi_collection.events and isi_collection.detector_operation_id == detector_id

    # The recording may now be closed; ISI consumes only the approved event table.
    data_store.unregister(registration.session_handle)
    _definition, artifact_store, isi_runner = _setup(tmp_path)
    result, _ = _run(
        isi_collection,
        ISIParameters(polarity_mode="separate"),
        isi_runner,
    )
    assert result.status is ExecutionStatus.SUCCEEDED, result.error
    assert result.output.detection_reference == f"execution://{detector_record.execution_id}"
    assert result.output.detector_execution_id == detector_record.execution_id
    assert result.output.detector_operation_id == detector_id
    artifact_store.open(result.output.interval_artifact_reference).close()


def test_consecutive_intervals_sort_per_channel_and_preserve_event_ids(tmp_path):
    collection = _collection(
        [
            _event("a10", 0, 10),
            _event("a2", 0, 2),
            _event("a5", 0, 5),
            _event("b20", 1, 20, "positive"),
            _event("b12", 1, 12, "positive"),
        ]
    )
    _definition, store, runner = _setup(tmp_path)
    result, _ = _run(
        collection,
        ISIParameters(polarity_mode="separate", short_interval_threshold_samples=5),
        runner,
    )
    assert result.status is ExecutionStatus.SUCCEEDED, result.error
    output = result.output
    assert output.interval_count == 3
    summaries = {(item.channel_index, item.polarity_group): item for item in output.summaries}
    channel_zero = summaries[(0, "negative")]
    assert channel_zero.detected_event_count == 3
    assert channel_zero.interval_count == 2
    assert channel_zero.mean_isi_samples == 4.0
    assert channel_zero.median_isi_samples == 4.0
    assert channel_zero.standard_deviation_isi_samples == 1.0
    assert channel_zero.minimum_isi_samples == 3
    assert channel_zero.maximum_isi_samples == 5
    assert channel_zero.coefficient_of_variation == 0.25
    assert channel_zero.intervals_below_threshold_count == 1
    assert channel_zero.short_interval_fraction == 0.5
    with store.open(output.interval_artifact_reference) as artifact:
        assert artifact.manifest["event_count"] == 5
        assert artifact.manifest["interval_count"] == 3
        rows = list(artifact.iter_rows())
    assert [(row["starting_event_id"], row["ending_event_id"], row["delta_samples"])
            for row in rows] == [("a2", "a5", 3), ("a5", "a10", 5), ("b12", "b20", 8)]
    assert rows[0]["delta_ms"] == 3.0
    assert rows[0]["channel_index"] == 0 and rows[2]["channel_index"] == 1


def test_polarity_separate_and_combine_have_explicit_different_groups(tmp_path):
    collection = _collection(
        [
            _event("p1", 0, 1, "positive"),
            _event("n1", 0, 2, "negative"),
            _event("p2", 0, 4, "positive"),
            _event("n2", 0, 10, "negative"),
        ]
    )
    _definition, store, runner = _setup(tmp_path)
    separate, _ = _run(collection, ISIParameters(polarity_mode="separate"), runner)
    combined, _ = _run(collection, ISIParameters(polarity_mode="combine"), runner)
    assert {item.polarity_group for item in separate.output.summaries} == {"positive", "negative"}
    assert [item.interval_count for item in separate.output.summaries] == [1, 1]
    assert combined.output.summaries[0].polarity_group == "combined"
    with store.open(combined.output.interval_artifact_reference) as artifact:
        assert [row["delta_samples"] for row in artifact.iter_rows()] == [1, 2, 6]


def test_duplicate_indices_are_kept_warned_and_report_zero_intervals(tmp_path):
    collection = _collection(
        [_event("b", 0, 5), _event("a", 0, 5), _event("c", 0, 9)]
    )
    _definition, store, runner = _setup(tmp_path)
    result, prepared = _run(collection, ISIParameters(polarity_mode="combine"), runner)
    assert result.status is ExecutionStatus.SUCCEEDED
    summary = result.output.summaries[0]
    assert summary.duplicate_event_count == 1
    assert summary.zero_length_interval_count == 1
    assert result.output.duplicate_event_count == 1
    assert result.output.zero_length_interval_count == 1
    assert any("Duplicate event indices" in warning for warning in prepared.warnings)
    assert any("Duplicate event indices" in warning for warning in result.warnings)
    with store.open(result.output.interval_artifact_reference) as artifact:
        rows = list(artifact.iter_rows())
    assert [(row["starting_event_id"], row["ending_event_id"], row["delta_samples"])
            for row in rows] == [("a", "b", 0), ("b", "c", 4)]


@pytest.mark.parametrize("event_count", [0, 1])
def test_zero_and_one_event_have_undefined_distribution_statistics(tmp_path, event_count):
    events = [_event("one", 0, 7)][:event_count]
    collection = _collection(events)
    _definition, store, runner = _setup(tmp_path)
    result, _ = _run(
        collection,
        ISIParameters(polarity_mode="separate", short_interval_threshold_samples=4),
        runner,
    )
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.output.interval_count == 0
    summary = result.output.summaries[0]
    assert summary.detected_event_count == event_count
    assert summary.mean_isi_samples is None
    assert summary.median_isi_samples is None
    assert summary.standard_deviation_isi_samples is None
    assert summary.minimum_isi_samples is None
    assert summary.maximum_isi_samples is None
    assert summary.coefficient_of_variation is None
    assert summary.intervals_below_threshold_count == 0
    assert summary.short_interval_fraction is None
    with store.open(result.output.interval_artifact_reference) as artifact:
        assert list(artifact.iter_rows()) == []


def test_constant_and_variable_interval_population_statistics(tmp_path):
    collection = _collection(
        [_event(f"e{i}", 0, sample) for i, sample in enumerate((0, 2, 4, 6))]
    )
    _definition, _store, runner = _setup(tmp_path)
    result, _ = _run(collection, ISIParameters(polarity_mode="combine"), runner)
    summary = result.output.summaries[0]
    assert summary.mean_isi_samples == 2
    assert summary.median_isi_samples == 2
    assert summary.standard_deviation_isi_samples == 0
    assert summary.coefficient_of_variation == 0

    variable = _collection([_event("x", 0, 1), _event("y", 0, 2), _event("z", 0, 5)])
    varied, _ = _run(variable, ISIParameters(polarity_mode="combine"), runner)
    summary = varied.output.summaries[0]
    assert summary.mean_isi_samples == 2
    assert summary.standard_deviation_isi_samples == 1
    assert summary.coefficient_of_variation == 0.5


def test_exactly_two_events_produce_one_interval(tmp_path):
    collection = _collection([_event("first", 1, 11), _event("second", 1, 19)])
    _definition, store, runner = _setup(tmp_path)
    result, _ = _run(collection, ISIParameters(polarity_mode="combine"), runner)
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.output.event_count == 2
    assert result.output.interval_count == 1
    summary = result.output.summaries[0]
    assert summary.mean_isi_samples == 8
    assert summary.median_isi_samples == 8
    assert summary.standard_deviation_isi_samples == 0
    assert summary.minimum_isi_samples == summary.maximum_isi_samples == 8
    assert summary.coefficient_of_variation == 0
    with store.open(result.output.interval_artifact_reference) as artifact:
        assert next(iter(artifact.iter_rows()))["starting_event_id"] == "first"


@pytest.mark.parametrize("status", [MetadataStatus.UNKNOWN, MetadataStatus.INFERRED])
def test_unknown_or_inferred_sampling_rate_keeps_only_sample_units_and_warns(tmp_path, status):
    rate = None if status is MetadataStatus.UNKNOWN else 1000.0
    collection = _collection(
        [_event("a", 0, 2), _event("b", 0, 5)],
        rate=rate,
        rate_status=status,
    )
    _definition, store, runner = _setup(tmp_path)
    result, prepared = _run(collection, ISIParameters(polarity_mode="combine"), runner)
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.output.interval_time_unit == "unavailable"
    summary = result.output.summaries[0]
    assert summary.mean_isi_samples == 3
    assert summary.mean_isi_ms is None
    assert summary.standard_deviation_isi_ms is None
    assert any("Sampling rate is" in warning for warning in prepared.warnings)
    with store.open(result.output.interval_artifact_reference) as artifact:
        assert next(artifact.iter_rows())["delta_ms"] is None


def test_confirmed_sampling_rate_converts_to_ms_and_ms_threshold_is_strict(tmp_path):
    collection = _collection([_event("a", 0, 0), _event("b", 0, 3), _event("c", 0, 8)])
    _definition, store, runner = _setup(tmp_path)
    result, _ = _run(
        collection,
        ISIParameters(polarity_mode="combine", short_interval_threshold_ms=3.0),
        runner,
    )
    assert result.output.interval_time_unit == "milliseconds"
    summary = result.output.summaries[0]
    assert summary.mean_isi_samples == 4
    assert summary.mean_isi_ms == 4
    assert summary.standard_deviation_isi_ms == 1
    assert summary.minimum_isi_ms == 3
    assert summary.maximum_isi_ms == 5
    assert summary.intervals_below_threshold_count == 0  # 3 ms equality is not below.
    assert summary.short_interval_fraction == 0
    with store.open(result.output.interval_artifact_reference) as artifact:
        assert [row["delta_ms"] for row in artifact.iter_rows()] == [3.0, 5.0]


def test_ms_threshold_requires_confirmed_rate_and_dual_thresholds_are_rejected(tmp_path):
    collection = _collection(
        [_event("a", 0, 0), _event("b", 0, 1)],
        rate=1000.0,
        rate_status=MetadataStatus.INFERRED,
    )
    _definition, _store, runner = _setup(tmp_path)
    with pytest.raises(ValueError, match="requires a confirmed sampling rate"):
        runner.prepare(
            OPERATION_ID,
            OPERATION_VERSION,
            ISIInput(events=collection),
            ISIParameters(polarity_mode="combine", short_interval_threshold_ms=1.0),
            context=_context(collection),
        )
    with pytest.raises(ValueError, match="not both"):
        ISIParameters(
            polarity_mode="combine",
            short_interval_threshold_samples=1,
            short_interval_threshold_ms=1.0,
        )


def test_context_selection_must_match_event_source_before_approval(tmp_path):
    collection = _collection([_event("a", 0, 1), _event("b", 0, 2)])
    _definition, _store, runner = _setup(tmp_path)
    mismatched = _context(collection).model_copy(
        update={"sample_interval": SampleInterval(start=1, stop=10)}
    )
    with pytest.raises(ValueError, match="interval does not match"):
        runner.prepare(
            OPERATION_ID,
            OPERATION_VERSION,
            ISIInput(events=collection),
            ISIParameters(polarity_mode="separate"),
            context=mismatched,
        )


def test_no_threshold_marks_threshold_metrics_unavailable_and_zero_mean_cv_is_none(tmp_path):
    collection = _collection([_event("a", 0, 5), _event("b", 0, 5)])
    _definition, store, runner = _setup(tmp_path)
    result, prepared = _run(collection, ISIParameters(polarity_mode="combine"), runner)
    summary = result.output.summaries[0]
    assert summary.mean_isi_samples == 0
    assert summary.standard_deviation_isi_samples == 0
    assert summary.coefficient_of_variation is None
    assert summary.intervals_below_threshold_count is None
    assert summary.short_interval_fraction is None
    assert any("No short-interval threshold" in warning for warning in prepared.warnings)
    with store.open(result.output.interval_artifact_reference) as artifact:
        assert next(artifact.iter_rows())["delta_samples"] == 0


def test_approval_is_required_and_exact_event_provenance_is_recorded(tmp_path):
    collection = _collection([_event("a", 0, 2), _event("b", 0, 7)])
    _definition, store, runner = _setup(tmp_path)
    prepared = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        ISIInput(events=collection),
        ISIParameters(polarity_mode="separate"),
        context=_context(collection),
    )
    with pytest.raises(ApprovalRequiredError):
        runner.execute(prepared)
    approval = runner.approve(prepared, approved_by="researcher")
    record = runner.execute(approval)
    assert record.status is ExecutionStatus.SUCCEEDED
    assert record.operation_id == OPERATION_ID
    assert record.implementation_version == IMPLEMENTATION_VERSION
    assert record.provenance.approved_by == "researcher"
    assert record.data_references == ()  # ISI uses event coordinates, never signal access.
    assert record.input_data.events.detector_operation_id == collection.detector_operation_id
    assert record.input_data.events.detection_reference == collection.detection_reference
    assert record.output.dataset == collection.dataset
    assert record.output.interval_artifact_reference
    with store.open(record.output.interval_artifact_reference) as artifact:
        assert artifact.manifest["execution_id"] == record.execution_id
        assert artifact.manifest["detector_operation_id"] == collection.detector_operation_id
        assert artifact.manifest["detection_reference"] == collection.detection_reference


def test_large_event_table_streams_interval_rows_to_artifact(tmp_path):
    count = 10_000
    collection = _collection(
        [_event(f"e{index:05d}", 0, index) for index in range(count)],
        sample_count=count,
    )
    _definition, store, runner = _setup(tmp_path)
    result, _ = _run(collection, ISIParameters(polarity_mode="combine"), runner)
    assert result.status is ExecutionStatus.SUCCEEDED, result.error
    assert result.output.event_count == count
    assert result.output.interval_count == count - 1
    assert result.output.summaries[0].mean_isi_samples == 1
    with store.open(result.output.interval_artifact_reference) as artifact:
        assert artifact.manifest["interval_count"] == count - 1
        iterator = artifact.iter_rows()
        assert next(iterator)["delta_samples"] == 1
        last = None
        for last in iterator:
            pass
        assert last["ending_event_id"] == f"e{count - 1:05d}"


def test_invalid_event_reference_and_polarity_inputs_are_rejected():
    with pytest.raises(ValueError, match="outside its source sample interval"):
        _collection([_event("bad", 0, 100)], sample_count=100)
    with pytest.raises(ValueError, match="event channel"):
        _collection([_event("bad", 3, 1)])
    with pytest.raises(ValueError, match="polarity_mode"):
        ISIParameters.model_validate({})
    with pytest.raises(ValueError, match="not both"):
        ISIParameters(
            polarity_mode="separate",
            short_interval_threshold_samples=2,
            short_interval_threshold_ms=2,
        )

