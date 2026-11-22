from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from neuroscope.analysis.contracts import (
    MetadataSnapshot,
    OperationContext,
    SampleInterval,
)
from neuroscope.analysis.mua_operation import (
    CALC_MUA_OPERATION,
    IMPLEMENTATION_VERSION,
    OPERATION_ID,
    OPERATION_VERSION,
    CalcMUAInput,
    CalcMUAOutput,
    CalcMUAParameters,
    register_calc_mua,
)
from neuroscope.analysis.operations import (
    ApprovalRequiredError,
    ExecutionStatus,
    OperationRunner,
    OperationValidationError,
)
from neuroscope.analysis.registry import OperationRegistry
from neuroscope.data_access import DatasetStore
from neuroscope.domain import DatasetMetadata, DatasetProfile
from neuroscope.exploration import DatasetExploration, MetadataStatus
from neuroscope.memory.project_store import ProjectStore


def _values() -> np.ndarray:
    values = np.tile(np.asarray([0.1, -0.1]), 50).astype(np.float64)
    values[30] = 2.0
    values[50] = -2.0
    return values


def _setup(
    tmp_path: Path,
    *,
    filtering_description: str | None = None,
    filtering_status: MetadataStatus = MetadataStatus.UNKNOWN,
    sampling_rate_hz: float | None = 1000.0,
    sampling_rate_status: MetadataStatus = MetadataStatus.CONFIRMED,
):
    values = _values()
    profile = DatasetProfile.from_array(tmp_path / "calc-mua.npy", values)
    profile = profile.model_copy(
        update={
            "metadata": DatasetMetadata(
                sampling_rate_hz=sampling_rate_hz,
                units="mV",
                signal_kind="raw_voltage",
                filtering_description=filtering_description,
            ),
            "confirmed_fields": [
                "signal_kind",
                "units",
                *(["sampling_rate_hz"] if sampling_rate_status is MetadataStatus.CONFIRMED else []),
                *(["filtering_description"] if filtering_status is MetadataStatus.CONFIRMED else []),
            ],
            "inferred_fields": (
                ["sampling_rate_hz"]
                if sampling_rate_status is MetadataStatus.INFERRED
                else ["filtering_description"]
                if filtering_status is MetadataStatus.INFERRED
                else []
            ),
            "unknown_fields": [
                *([] if sampling_rate_status is not MetadataStatus.UNKNOWN else ["sampling_rate_hz"]),
                *([] if filtering_status is not MetadataStatus.UNKNOWN else ["filtering_description"]),
            ],
        }
    )
    exploration = DatasetExploration(values, profile)
    store = DatasetStore()
    registration = store.register(exploration)
    reference = registration.reference((0,), SampleInterval(start=20, stop=70))
    registry = OperationRegistry()
    definition = register_calc_mua(registry)
    project = ProjectStore(tmp_path / "project")
    runner = OperationRunner(
        registry,
        data_store=store,
        mua_artifact_store=project.mua_artifacts,
    )
    fields = (
        ("signal_kind", "raw_voltage", MetadataStatus.CONFIRMED),
        ("sampling_rate_hz", sampling_rate_hz, sampling_rate_status),
        ("units", "mV", MetadataStatus.CONFIRMED),
        ("filtering_description", filtering_description, filtering_status),
    )
    context = OperationContext(
        dataset_reference=registration.dataset,
        selected_channels=(0,),
        sample_interval=reference.sample_interval,
        metadata=tuple(
            MetadataSnapshot(name=name, value=value, status=status)
            for name, value, status in fields
        ),
    )
    parameters = CalcMUAParameters(
        stdmin=1.0,
        mua_window_s=0.003,
        mode="rel",
        num_points=10,
    )
    return values, store, registration, reference, registry, definition, project, runner, context, parameters


def test_operation_registration_and_explicit_parameters():
    registry = OperationRegistry()
    definition = register_calc_mua(registry)
    assert definition is CALC_MUA_OPERATION
    assert registry.get(OPERATION_ID, version=OPERATION_VERSION) is definition
    assert definition.implementation_version == IMPLEMENTATION_VERSION
    assert definition.input_model is CalcMUAInput
    assert definition.parameters_model is CalcMUAParameters
    assert definition.output_model is CalcMUAOutput
    with pytest.raises(ValueError, match="already registered"):
        register_calc_mua(registry)
    with pytest.raises(ValidationError):
        CalcMUAParameters(stdmin=1, mua_window_s=0, mode="rel", num_points=10)
    with pytest.raises(ValidationError):
        CalcMUAParameters(stdmin=1, mua_window_s=0.1, mode="rel")


def test_prepare_is_side_effect_free_then_approval_writes_project_artifact(tmp_path, monkeypatch):
    values, store, _registration, reference, _registry, _definition, project, runner, context, parameters = _setup(
        tmp_path
    )
    original = values.copy()
    real_resolve = store.resolve
    calls = []

    def tracked_resolve(data_reference):
        calls.append(data_reference)
        return real_resolve(data_reference)

    monkeypatch.setattr(store, "resolve", tracked_resolve)
    prepared = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        CalcMUAInput(signal=reference),
        parameters,
        context=context,
    )
    assert len(prepared.warnings) == 1
    assert "does not filter" in prepared.warnings[0]
    assert calls == []
    assert not list(project.mua_artifacts.root.glob("*/manifest.json"))
    with pytest.raises(ApprovalRequiredError):
        runner.execute(prepared)

    approved = runner.approve(prepared, approved_by="Researcher")
    assert calls == []
    record = runner.execute(approved)
    assert record.status is ExecutionStatus.SUCCEEDED
    assert calls == [reference]
    assert np.array_equal(values, original)
    output = record.output
    assert isinstance(output, CalcMUAOutput)
    assert output.sample_interval == reference.sample_interval
    assert output.source_channel == 0
    assert output.output_length == 50
    assert output.event_count == 1
    assert output.threshold == pytest.approx(0.1 / 0.6745)
    assert output.mua_units == "s^-1"
    assert output.amplitude_units == "mV"
    assert output.filtering_status is MetadataStatus.UNKNOWN
    assert record.execution_id == output.execution_id
    assert record.implementation_version == IMPLEMENTATION_VERSION
    assert any("filtering is not confirmed" in warning for warning in record.warnings)
    snapshot = record.to_snapshot()
    assert "mua" not in snapshot["output"]
    artifact_reference = output.artifact_reference
    with project.mua_artifacts.open(artifact_reference) as artifact:
        assert artifact.mua.shape == (50,)
        assert artifact.spike_sample_indices.tolist() == [30]
        assert artifact.spike_times_s.tolist() == pytest.approx([0.011])
        assert artifact.manifest["sample_interval"] == {"start": 20, "stop": 70}
        assert artifact.manifest["execution_id"] == record.execution_id
        assert artifact.manifest["parameters"]["num_points"] == 10
        with pytest.raises(ValueError):
            artifact.mua[0] = 1


def test_confirmed_filter_metadata_is_preserved_without_warning(tmp_path):
    (
        _values_array,
        _store,
        _registration,
        reference,
        _registry,
        _definition,
        _project,
        runner,
        _context,
        parameters,
    ) = _setup(
        tmp_path,
        filtering_description="explicit test high-pass preprocessing",
        filtering_status=MetadataStatus.CONFIRMED,
    )
    fields = (
        MetadataSnapshot(name="signal_kind", value="raw_voltage", status=MetadataStatus.CONFIRMED),
        MetadataSnapshot(name="sampling_rate_hz", value=1000.0, status=MetadataStatus.CONFIRMED),
        MetadataSnapshot(name="units", value="mV", status=MetadataStatus.CONFIRMED),
        MetadataSnapshot(
            name="filtering_description",
            value="explicit test high-pass preprocessing",
            status=MetadataStatus.CONFIRMED,
        ),
    )
    context = OperationContext(
        dataset_reference=reference.dataset,
        selected_channels=(0,),
        sample_interval=reference.sample_interval,
        metadata=fields,
    )
    prepared = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        {"signal": reference},
        parameters,
        context=context,
    )
    assert not prepared.warnings
    record = runner.execute(runner.approve(prepared, approved_by="Researcher"))
    assert record.output.filtering_description == "explicit test high-pass preprocessing"
    assert record.output.filtering_status is MetadataStatus.CONFIRMED


@pytest.mark.parametrize(
    "context_update",
    [
        {"selected_channels": (1,)},
        {"sample_interval": SampleInterval(start=0, stop=50)},
    ],
)
def test_operation_context_must_match_approved_data_reference(tmp_path, context_update):
    (
        _values_array,
        _store,
        _registration,
        reference,
        _registry,
        _definition,
        _project,
        runner,
        context,
        parameters,
    ) = _setup(tmp_path)
    mismatched = context.model_copy(update=context_update)
    with pytest.raises(ValueError, match="context"):
        runner.prepare(
            OPERATION_ID,
            OPERATION_VERSION,
            {"signal": reference},
            parameters,
            context=mismatched,
        )


@pytest.mark.parametrize(
    "rate,status",
    [(None, MetadataStatus.UNKNOWN), (1000.0, MetadataStatus.INFERRED)],
)
def test_sampling_rate_must_be_confirmed(tmp_path, rate, status):
    (
        _values_array,
        _store,
        _registration,
        reference,
        _registry,
        _definition,
        _project,
        runner,
        _context,
        parameters,
    ) = _setup(tmp_path, sampling_rate_hz=rate, sampling_rate_status=status)
    context = OperationContext(
        dataset_reference=reference.dataset,
        selected_channels=(0,),
        sample_interval=reference.sample_interval,
        metadata=(
            MetadataSnapshot(name="signal_kind", value="raw_voltage", status=MetadataStatus.CONFIRMED),
            MetadataSnapshot(name="sampling_rate_hz", value=rate, status=status),
            MetadataSnapshot(name="units", value="mV", status=MetadataStatus.CONFIRMED),
            MetadataSnapshot(name="filtering_description", value=None, status=MetadataStatus.UNKNOWN),
        ),
    )
    with pytest.raises(OperationValidationError, match="sampling_rate_hz"):
        runner.prepare(
            OPERATION_ID,
            OPERATION_VERSION,
            {"signal": reference},
            parameters,
            context=context,
        )


def test_operation_reads_only_the_approved_lazy_sample_interval(tmp_path):
    class LazySamples:
        def __init__(self, data):
            self.data = data
            self.shape = data.shape
            self.dtype = data.dtype
            self.reads = []
            self.closed = False

        def read_window(self, start, stop, *, channel_axis=None, channel_index=None):
            self.reads.append((start, stop, channel_index))
            values = self.data[start:stop].copy()
            values.setflags(write=False)
            return values

        def close(self):
            self.closed = True

    values = _values()
    lazy = LazySamples(values)
    profile = DatasetProfile.from_array(tmp_path / "lazy-calc-mua.mat", values)
    profile = profile.model_copy(
        update={
            "metadata": DatasetMetadata(
                sampling_rate_hz=1000.0,
                signal_kind="raw_voltage",
            ),
            "confirmed_fields": ["sampling_rate_hz", "signal_kind"],
            "inferred_fields": [],
            "unknown_fields": ["units", "filtering_description"],
        }
    )
    exploration = DatasetExploration(lazy, profile)
    data_store = DatasetStore()
    registration = data_store.register(exploration)
    reference = registration.reference((0,), SampleInterval(start=20, stop=70))
    registry = OperationRegistry()
    register_calc_mua(registry)
    project = ProjectStore(tmp_path / "lazy-project")
    runner = OperationRunner(
        registry,
        data_store=data_store,
        mua_artifact_store=project.mua_artifacts,
    )
    context = OperationContext(
        dataset_reference=registration.dataset,
        selected_channels=(0,),
        sample_interval=reference.sample_interval,
        metadata=(
            MetadataSnapshot(name="signal_kind", value="raw_voltage", status=MetadataStatus.CONFIRMED),
            MetadataSnapshot(name="sampling_rate_hz", value=1000.0, status=MetadataStatus.CONFIRMED),
            MetadataSnapshot(name="units", value=None, status=MetadataStatus.UNKNOWN),
            MetadataSnapshot(name="filtering_description", value=None, status=MetadataStatus.UNKNOWN),
        ),
    )
    prepared = runner.prepare(
        OPERATION_ID,
        OPERATION_VERSION,
        {"signal": reference},
        CalcMUAParameters(stdmin=1, mua_window_s=0.003, mode="rel", num_points=7),
        context=context,
    )
    assert lazy.reads == []
    result = runner.execute(runner.approve(prepared, approved_by="Researcher"))
    assert result.status is ExecutionStatus.SUCCEEDED
    assert lazy.reads == [(20, 70, 0)]
    assert result.output.sample_interval == reference.sample_interval
    data_store.unregister(registration.session_handle)
    assert lazy.closed
