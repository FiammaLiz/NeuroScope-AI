from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from pydantic import Field

from neuroscope.analysis.contracts import (
    DataReference,
    DatasetReference,
    OperationDefinition,
    OperationModel,
    OperationResult,
    SampleInterval,
)
from neuroscope.analysis.isi_operation import register_isi_analysis
from neuroscope.analysis.mua_average_operation import (
    MUAArtifactReference,
    register_mua_average,
)
from neuroscope.analysis.mua_operation import register_calc_mua
from neuroscope.analysis.operations import ApprovalRequiredError, OperationRunner
from neuroscope.analysis.pipelines import (
    PipelineExecutionService,
    PipelineInputBinding,
    PipelineNode,
    new_pipeline,
    validate_pipeline,
)
from neuroscope.analysis.registry import OperationRegistry
from neuroscope.analysis.signed_peak_operation import register_signed_peak_detector
from neuroscope.analysis.spike_operation import register_negative_spike_detector
from neuroscope.analysis.waveform_operation import register_waveform_extraction
from neuroscope.data_access import DatasetStore
from neuroscope.domain import DatasetMetadata, DatasetProfile
from neuroscope.exploration import DatasetExploration, MetadataStatus
from neuroscope.memory.project_store import ProjectStore


class _SignalInput(OperationModel):
    signal: DataReference


class _Parameters(OperationModel):
    multiplier: int = Field(strict=True, gt=0)


class _Output(OperationModel):
    value: int = Field(strict=True)


def _fixture(tmp_path):
    values = np.arange(16, dtype=np.float32)
    profile = DatasetProfile.from_array(Path("pipeline-fixture.npy"), values)
    exploration = DatasetExploration(values, profile)
    data_store = DatasetStore()
    registration = data_store.register(exploration)
    calls = []

    def execute(input_data, parameters, context, runtime):
        calls.append(input_data.signal.sample_interval)
        samples = runtime.data_for(input_data.signal).channel(0).samples
        return OperationResult(_Output(value=int(samples.sum()) * parameters.multiplier))

    operation = OperationDefinition(
        operation_id="neuroscope.test.pipeline-probe",
        name="Pipeline test probe",
        description="Deterministic test operation for pipeline lifecycle tests.",
        version="1.0.0",
        implementation_version="test-v1",
        input_model=_SignalInput,
        parameters_model=_Parameters,
        output_model=_Output,
        execute=execute,
    )
    registry = OperationRegistry()
    registry.register(operation)
    runner = OperationRunner(registry, data_store=data_store)
    service = PipelineExecutionService(registry, runner, data_store)
    binding = PipelineInputBinding(
        target_field="signal",
        source_kind="dataset",
        dataset=registration.dataset,
        channels=(0,),
        sample_interval=SampleInterval(start=2, stop=6),
    )
    node = PipelineNode(
        node_id="measure",
        operation_id=operation.operation_id,
        operation_version=operation.version,
        parameters={"multiplier": 2},
        inputs=(binding,),
    )
    pipeline = new_pipeline("Test pipeline", (node,))
    return values, data_store, registry, runner, service, pipeline, calls


def test_pipeline_durable_dataset_binding_contains_no_session_handle(tmp_path):
    _, _, registry, _, _, pipeline, _ = _fixture(tmp_path)
    serialized = pipeline.model_dump_json()

    assert "session_handle" not in serialized
    assert validate_pipeline(pipeline, registry).structurally_valid
    assert not validate_pipeline(pipeline, registry).execution_ready


def test_pipeline_validation_rejects_cycle_unknown_operation_and_bad_parameters(tmp_path):
    _, _, registry, _, _, pipeline, _ = _fixture(tmp_path)
    first, = pipeline.nodes
    cyclic = pipeline.model_copy(update={"nodes": (
        first.model_copy(update={"depends_on": ("second",)}),
        first.model_copy(update={"node_id": "second", "depends_on": ("measure",)}),
    )})
    assert not validate_pipeline(cyclic, registry).structurally_valid
    assert any("cycle" in item for item in validate_pipeline(cyclic, registry).errors)

    unknown = pipeline.model_copy(update={"nodes": (
        first.model_copy(update={"operation_id": "neuroscope.unknown.op"}),
    )})
    assert not validate_pipeline(unknown, registry).structurally_valid

    bad = pipeline.model_copy(update={"nodes": (
        first.model_copy(update={"parameters": {"multiplier": 0}}),
    )})
    assert not validate_pipeline(bad, registry).structurally_valid


def test_pipeline_validation_distinguishes_missing_dataset_from_invalid_graph(tmp_path):
    _, data_store, registry, _, _, pipeline, _ = _fixture(tmp_path)
    report = validate_pipeline(pipeline, registry)

    assert report.structurally_valid
    assert not report.execution_ready
    assert report.nodes[0].status == "blocked"
    active = validate_pipeline(pipeline, registry, dataset_store=data_store)
    assert active.execution_ready
    assert active.nodes[0].status == "ready"

    scientific_registry = OperationRegistry()
    register_negative_spike_detector(scientific_registry)
    original_node = pipeline.nodes[0]
    detector_node = original_node.model_copy(update={
        "operation_id": "neuroscope.spike.detect-negative",
        "parameters": {"threshold_sd": -4.0},
    })
    metadata_blocked = validate_pipeline(
        new_pipeline("Unknown signal kind", (detector_node,)),
        scientific_registry,
        dataset_store=data_store,
    )
    assert metadata_blocked.structurally_valid
    assert not metadata_blocked.execution_ready
    assert metadata_blocked.nodes[0].status == "blocked"
    assert any("signal_kind" in reason for reason in metadata_blocked.nodes[0].reasons)


def test_pipeline_node_requires_explicit_approval_and_executes_through_runner(tmp_path):
    original, data_store, _registry, runner, service, pipeline, calls = _fixture(tmp_path)
    state = service.start(pipeline)
    plan, prepared = service.prepare_node(
        pipeline, state, "measure", node_records={}
    )
    assert calls == []
    assert plan.operation_id == "neuroscope.test.pipeline-probe"
    assert prepared.operation_id == plan.operation_id
    with pytest.raises(ApprovalRequiredError):
        runner.execute(prepared)  # type: ignore[arg-type]
    assert calls == []

    approved = service.approve_node(state.run_id, "measure", approved_by="Researcher")
    assert calls == []
    next_state, record = service.execute_node(pipeline, state, "measure", approved)

    assert calls == [SampleInterval(start=2, stop=6)]
    assert record.output.value == 28
    assert next_state.node_status["measure"] == "succeeded"
    assert record.data_references[0].sample_interval == SampleInterval(start=2, stop=6)
    np.testing.assert_array_equal(original, np.arange(16, dtype=np.float32))
    data_store.unregister(record.data_references[0].session_handle)


def test_pipeline_input_change_after_prepare_invalidates_pipeline_review(tmp_path):
    _, _, _, _, service, pipeline, _ = _fixture(tmp_path)
    state = service.start(pipeline)
    service.prepare_node(pipeline, state, "measure", node_records={})
    approved = service.approve_node(state.run_id, "measure", approved_by="Researcher")
    changed_node = pipeline.nodes[0].model_copy(update={"parameters": {"multiplier": 3}})
    changed_pipeline = pipeline.model_copy(update={"nodes": (changed_node,)})

    with pytest.raises(ValueError, match="changed after preparation"):
        service.execute_node(changed_pipeline, state, "measure", approved)


def test_pipeline_revision_persistence_is_non_overwriting_and_reopenable(tmp_path):
    _, _, _, _, _, pipeline, _ = _fixture(tmp_path)
    project = ProjectStore(tmp_path / "project")
    saved_path = project.save_pipeline(pipeline)

    reopened = project.load_pipeline(pipeline.pipeline_id)
    assert reopened == pipeline
    assert project.list_pipelines()[0]["status"] == "available"
    reopened_project = ProjectStore.open_existing(tmp_path / "project")
    assert reopened_project.load_pipeline(pipeline.pipeline_id) == pipeline
    assert reopened_project.load_execution_history() == []
    with pytest.raises(FileExistsError):
        project.save_pipeline(pipeline)
    assert saved_path.is_file()
    revised = project.save_pipeline_revision(pipeline)
    assert project.load_pipeline(pipeline.pipeline_id).revision == 2
    assert revised.is_file()
    state = PipelineExecutionService(
        OperationRegistry(), OperationRunner(OperationRegistry()), DatasetStore()
    ).start(pipeline)
    project.save_pipeline_run(state)
    assert project.list_pipeline_runs()[0]["run"].run_id == state.run_id


def test_dataset_replacement_invalidates_a_prepared_pipeline_reference(tmp_path):
    original = np.arange(16, dtype=np.float32)
    profile = DatasetProfile.from_array(Path("stale-pipeline.npy"), original)
    profile = profile.model_copy(update={
        "metadata": DatasetMetadata(signal_kind="raw_voltage"),
        "confirmed_fields": ["signal_kind"],
        "inferred_fields": [],
        "unknown_fields": ["sampling_rate_hz", "units", "filtering_description"],
    })
    data_store = DatasetStore()
    registration = data_store.register(DatasetExploration(original, profile))
    signal = PipelineInputBinding(
        target_field="signal", source_kind="dataset", dataset=registration.dataset,
        channels=(0,), sample_interval=SampleInterval(start=2, stop=6),
    )
    detector = PipelineNode(
        node_id="detect", operation_id="neuroscope.spike.detect-negative",
        operation_version="1.0.0", parameters={"threshold_sd": -4.0}, inputs=(signal,),
    )
    waveform = PipelineNode(
        node_id="waveforms", operation_id="neuroscope.spike.extract-waveforms",
        operation_version="1.0.0", parameters={"pre_samples": 1, "post_samples": 1},
        inputs=(signal, PipelineInputBinding(
            target_field="detections", source_kind="node_output", source_node_id="detect",
            source_port="detection_events",
        )), depends_on=("detect",),
    )
    pipeline = new_pipeline("Stale registration pipeline", (detector, waveform))
    registry = OperationRegistry()
    register_negative_spike_detector(registry)
    register_waveform_extraction(registry)
    runner = OperationRunner(registry, data_store=data_store)
    service = PipelineExecutionService(registry, runner, data_store)
    state = service.start(pipeline)
    _plan, prepared = service.prepare_node(pipeline, state, "detect", node_records={})
    approved = service.approve_node(state.run_id, "detect", approved_by="Researcher")
    old_reference = prepared.input_data.signal
    data_store.unregister(old_reference.session_handle)
    replacement = np.full(16, 9, dtype=np.float32)
    profile = DatasetProfile.from_array(Path("pipeline-fixture.npy"), replacement)
    data_store.register(DatasetExploration(replacement, profile))

    next_state, record = service.execute_node(pipeline, state, "detect", approved)

    assert record.status.value == "failed"
    assert "stale dataset handle" in record.error.message
    assert next_state.node_status["waveforms"] == "blocked"
    with pytest.raises(ValueError, match="incomplete dependencies"):
        service.prepare_node(pipeline, next_state, "waveforms", node_records={"detect": record})
    np.testing.assert_array_equal(original, np.arange(16, dtype=np.float32))


def test_historical_execution_reuse_preserves_identity_without_rerunning(tmp_path):
    _, _store, registry, _runner, service, pipeline, calls = _fixture(tmp_path)
    state = service.start(pipeline)
    _plan, _prepared = service.prepare_node(pipeline, state, "measure", node_records={})
    approved = service.approve_node(state.run_id, "measure", approved_by="Researcher")
    state, record = service.execute_node(pipeline, state, "measure", approved)
    assert record.status.value == "succeeded"
    assert len(calls) == 1

    reused_node = PipelineNode(
        node_id="prior-result",
        operation_id=record.operation_id,
        operation_version=record.operation_version,
        reuse_execution_id=record.execution_id,
    )
    reused_pipeline = new_pipeline("Reuse result", (reused_node,))
    report = validate_pipeline(
        reused_pipeline, registry, execution_records={record.execution_id: record}
    )
    assert report.nodes[0].status == "reused"
    reused_state = service.start(reused_pipeline)
    reused_state = service.bind_reused_nodes(
        reused_pipeline, reused_state, {record.execution_id: record}
    )
    assert reused_state.execution_ids["prior-result"] == record.execution_id
    with pytest.raises(ValueError, match="do not prepare"):
        service.prepare_node(
            reused_pipeline, reused_state, "prior-result", node_records={},
            historical_records={record.execution_id: record},
        )
    assert len(calls) == 1


def test_validator_accepts_only_the_audited_scientific_operation_connections():
    registry = OperationRegistry()
    register_negative_spike_detector(registry)
    register_signed_peak_detector(registry)
    register_waveform_extraction(registry)
    register_isi_analysis(registry)
    register_calc_mua(registry)
    register_mua_average(registry)
    dataset = DatasetReference(dataset_id="synthetic-recording")
    signal = PipelineInputBinding(
        target_field="signal", source_kind="dataset", dataset=dataset,
        channels=(0,), sample_interval=SampleInterval(start=0, stop=100),
    )
    detect = PipelineNode(
        node_id="detect", operation_id="neuroscope.spike.detect-negative",
        operation_version="1.0.0", parameters={"threshold_sd": -4.0}, inputs=(signal,),
    )
    event_binding_wave = PipelineInputBinding(
        target_field="detections", source_kind="node_output",
        source_node_id="detect", source_port="detection_events",
    )
    wave = PipelineNode(
        node_id="waveforms", operation_id="neuroscope.spike.extract-waveforms",
        operation_version="1.0.0", parameters={"pre_samples": 2, "post_samples": 3},
        inputs=(signal, event_binding_wave), depends_on=("detect",),
    )
    event_binding_isi = PipelineInputBinding(
        target_field="events", source_kind="node_output",
        source_node_id="detect", source_port="detection_events",
    )
    isi = PipelineNode(
        node_id="isi", operation_id="neuroscope.spike.analyze-isi",
        operation_version="1.0.0", parameters={"polarity_mode": "separate"},
        inputs=(event_binding_isi,), depends_on=("detect",),
    )
    calc_nodes = []
    for node_id, channel in (("mua_a", 0), ("mua_b", 1)):
        channel_binding = signal.model_copy(update={"channels": (channel,)})
        calc_nodes.append(PipelineNode(
            node_id=node_id, operation_id="neuroscope.mua.calc-mua",
            operation_version="1.0.0",
            parameters={"stdmin": -4.0, "mua_window_s": 0.005, "mode": "rel", "num_points": 100},
            inputs=(channel_binding,),
        ))
    average = PipelineNode(
        node_id="mean", operation_id="neuroscope.mua.average-traces",
        operation_version="1.0.0", parameters={},
        inputs=(
            PipelineInputBinding(target_field="traces", source_kind="node_output", source_node_id="mua_a", source_port="mua_trace"),
            PipelineInputBinding(target_field="traces", source_kind="node_output", source_node_id="mua_b", source_port="mua_trace"),
        ), depends_on=("mua_a", "mua_b"),
    )
    pipeline = new_pipeline("Supported paths", (detect, wave, isi, *calc_nodes, average))

    report = validate_pipeline(pipeline, registry)

    assert report.structurally_valid
    assert set(report.topological_order) == {"detect", "waveforms", "isi", "mua_a", "mua_b", "mean"}
    assert all(item.status == "blocked" for item in report.nodes), report.nodes


def test_validator_rejects_detector_to_calc_mua_connection():
    registry = OperationRegistry()
    register_negative_spike_detector(registry)
    register_calc_mua(registry)
    dataset = DatasetReference(dataset_id="synthetic-recording")
    dataset_binding = PipelineInputBinding(
        target_field="signal", source_kind="dataset", dataset=dataset,
        channels=(0,), sample_interval=SampleInterval(start=0, stop=100),
    )
    detector = PipelineNode(
        node_id="detect", operation_id="neuroscope.spike.detect-negative",
        operation_version="1.0.0", parameters={"threshold_sd": -4.0}, inputs=(dataset_binding,),
    )
    unsupported = PipelineNode(
        node_id="mua", operation_id="neuroscope.mua.calc-mua", operation_version="1.0.0",
        parameters={"stdmin": -4.0, "mua_window_s": 0.005, "mode": "rel", "num_points": 100},
        inputs=(PipelineInputBinding(
            target_field="signal", source_kind="node_output", source_node_id="detect",
            source_port="detection_events",
        ),), depends_on=("detect",),
    )
    report = validate_pipeline(new_pipeline("Invalid edge", (detector, unsupported)), registry)
    assert not report.structurally_valid
    assert any("unsupported operation connection" in reason for item in report.nodes for reason in item.reasons)


def test_pipeline_executes_detection_then_waveform_and_isi_as_separate_approved_nodes(tmp_path):
    rng = np.random.default_rng(17)
    values = rng.normal(0.0, 0.1, 500).astype(np.float64)
    values[[100, 200, 300]] = -5.0
    profile = DatasetProfile.from_array(Path("pipeline-neural.npy"), values)
    profile = profile.model_copy(update={
        "metadata": DatasetMetadata(
            sampling_rate_hz=1000.0, units="mV", signal_kind="raw_voltage"
        ),
        "confirmed_fields": ["sampling_rate_hz", "units", "signal_kind"],
        "inferred_fields": [],
        "unknown_fields": [],
    })
    data_store = DatasetStore()
    registration = data_store.register(DatasetExploration(values, profile))
    interval = SampleInterval(start=0, stop=values.size)
    signal = PipelineInputBinding(
        target_field="signal", source_kind="dataset", dataset=registration.dataset,
        channels=(0,), sample_interval=interval,
    )
    detector = PipelineNode(
        node_id="detect", operation_id="neuroscope.spike.detect-negative",
        operation_version="1.0.0", parameters={"threshold_sd": -4.0}, inputs=(signal,),
    )
    waveforms = PipelineNode(
        node_id="waveforms", operation_id="neuroscope.spike.extract-waveforms",
        operation_version="1.0.0", parameters={"pre_samples": 2, "post_samples": 2},
        inputs=(signal, PipelineInputBinding(
            target_field="detections", source_kind="node_output", source_node_id="detect",
            source_port="detection_events",
        )), depends_on=("detect",),
    )
    isi = PipelineNode(
        node_id="isi", operation_id="neuroscope.spike.analyze-isi",
        operation_version="1.0.0", parameters={"polarity_mode": "separate"},
        inputs=(PipelineInputBinding(
            target_field="events", source_kind="node_output", source_node_id="detect",
            source_port="detection_events",
        ),), depends_on=("detect",),
    )
    pipeline = new_pipeline("Synthetic detector workflow", (detector, waveforms, isi))
    store = ProjectStore(tmp_path / "pipeline-project")
    registry = OperationRegistry()
    register_negative_spike_detector(registry)
    register_waveform_extraction(registry)
    register_isi_analysis(registry)
    runner = OperationRunner(
        registry, data_store=data_store, artifact_store=store.artifacts,
        isi_artifact_store=store.isi_artifacts,
    )
    service = PipelineExecutionService(registry, runner, data_store)
    assert validate_pipeline(pipeline, registry, dataset_store=data_store).structurally_valid
    state = service.start(pipeline)
    records_by_node = {}
    for node_id in ("detect", "waveforms", "isi"):
        _plan, _prepared = service.prepare_node(
            pipeline, state, node_id, node_records=records_by_node
        )
        approved = service.approve_node(state.run_id, node_id, approved_by="Researcher")
        state, record = service.execute_node(
            pipeline, state, node_id, approved, project_store=store
        )
        assert record.status.value == "succeeded", record.error
        records_by_node[node_id] = record
    assert len(records_by_node["detect"].output.original_sample_indices) >= 2
    assert records_by_node["waveforms"].output.accepted_event_count >= 2
    assert records_by_node["isi"].output.event_count >= 2
    assert all(status == "succeeded" for status in state.node_status.values())
    assert len(store.load_execution_history()) == 3


def test_pipeline_selected_mua_artifacts_are_typed_and_compatibility_checked():
    registry = OperationRegistry()
    register_mua_average(registry)
    interval = SampleInterval(start=0, stop=1000)

    def artifact(channel):
        return MUAArtifactReference(
            artifact_reference=f"artifact://mua/{uuid4()}",
            execution_id=f"exec-{channel}",
            dataset_id="same-recording",
            channel_index=channel,
            sample_interval=interval,
            output_length=100,
            sampling_rate_hz=1000.0,
            sampling_rate_status=MetadataStatus.CONFIRMED,
            mua_units="s^-1",
            amplitude_units="mV",
            amplitude_units_status=MetadataStatus.CONFIRMED,
            mode="rel",
            stdmin=-4.0,
            actual_threshold=-5.0,
            noise_estimate=1.25,
            bandwidth_s=0.005,
            num_points=100,
            source_sha256="source-config-v1",
            source_operation_id="neuroscope.mua.calc-mua",
            source_operation_version="1.0.0",
            source_implementation_version="matlab-calcMUA-v1",
            signal_kind="raw_voltage",
            signal_kind_status=MetadataStatus.CONFIRMED,
            filtering_description="No preprocessing applied in the explicit source setup.",
            filtering_status=MetadataStatus.CONFIRMED,
        )

    first, second = artifact(0), artifact(1)
    node = PipelineNode(
        node_id="mean",
        operation_id="neuroscope.mua.average-traces",
        operation_version="1.0.0",
        parameters={},
        inputs=tuple(
            PipelineInputBinding(
                target_field="traces", source_kind="artifact",
                artifact_value=item.model_dump(mode="json"),
            )
            for item in (first, second)
        ),
    )
    pipeline = new_pipeline("Saved MUA average", (node,))
    available = {first.artifact_reference, second.artifact_reference}
    report = validate_pipeline(
        pipeline, registry, available_artifact_references=available
    )
    assert report.execution_ready
    assert report.nodes[0].status == "ready"

    runner = OperationRunner(registry)
    service = PipelineExecutionService(registry, runner, DatasetStore())
    state = service.start(pipeline)
    plan, prepared = service.prepare_node(
        pipeline, state, "mean", node_records={}
    )
    assert plan.operation_id == "neuroscope.mua.average-traces"
    assert {item.artifact_reference for item in prepared.input_data.traces} == available

