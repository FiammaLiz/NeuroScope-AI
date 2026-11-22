from datetime import timezone

import numpy as np
import pytest
from pydantic import ConfigDict, Field, ValidationError

from neuroscope.analysis.contracts import (
    DatasetReference,
    ExecutionStatus,
    MetadataRequirement,
    MetadataSnapshot,
    OperationContext,
    OperationDefinition,
    OperationModel,
    OperationResult,
    PrerequisiteReference,
    ReproducibilityInfo,
    SampleInterval,
)
from neuroscope.analysis.operations import (
    ApprovalMismatchError,
    ApprovalRequiredError,
    OperationRunner,
    OperationValidationError,
)
from neuroscope.analysis.registry import AnalysisRegistry, OperationRegistry
from neuroscope.exploration import MetadataStatus


class ScalarInput(OperationModel):
    value: int = Field(strict=True)


class ArrayInput(OperationModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    samples: np.ndarray


class ArrayOutput(OperationModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    samples: np.ndarray


class ScaleParameters(OperationModel):
    factor: int = Field(gt=0, strict=True)


class ScalarOutput(OperationModel):
    value: int = Field(strict=True)


def _context(
    *,
    dataset_id="dataset-a",
    rate=1000.0,
    rate_status=MetadataStatus.CONFIRMED,
    prerequisite_execution="parent-run-a",
):
    return OperationContext(
        dataset_reference=DatasetReference(
            dataset_id=dataset_id,
            source_reference="synthetic://recording-a",
            content_fingerprint="sha256:abc123",
        ),
        metadata=(
            MetadataSnapshot(
                name="sampling_rate_hz", value=rate, status=rate_status
            ),
            MetadataSnapshot(name="units", value=None, status=MetadataStatus.UNKNOWN),
        ),
        selected_channels=(1,),
        sample_interval=SampleInterval(start=20, stop=80),
        prerequisites=(
            PrerequisiteReference(
                execution_id=prerequisite_execution,
                operation_id="neuroscope.test.parent",
                operation_version="1.2.0",
                output_reference="artifact://parent-output",
                status=ExecutionStatus.SUCCEEDED,
            ),
        ),
        reproducibility=ReproducibilityInfo(
            seed=31,
            rng_algorithm="PCG64",
            configuration_reference="synthetic-config://config-31",
        ),
    )


def _definition(executor=None, *, implementation_version="0.4.1"):
    calls = []

    def execute(input_data, parameters, context):
        calls.append((input_data.value, parameters.factor))
        if executor is not None:
            return executor(input_data, parameters, context)
        return OperationResult(
            ScalarOutput(value=input_data.value * parameters.factor),
            output_metadata=(
                MetadataSnapshot(
                    name="units", value=None, status=MetadataStatus.UNKNOWN
                ),
            ),
            warnings=("test execution warning",),
        )

    operation = OperationDefinition(
        operation_id="neuroscope.test.scale",
        name="Test scalar scaling",
        description="Test-only deterministic operation for exercising the runner.",
        version="1.3.0",
        implementation_version=implementation_version,
        input_model=ScalarInput,
        parameters_model=ScaleParameters,
        output_model=ScalarOutput,
        execute=execute,
        metadata_requirements=(
            MetadataRequirement(
                name="sampling_rate_hz", required=True, require_confirmed=True
            ),
            MetadataRequirement(name="units", required=False),
        ),
        prerequisites=("neuroscope.test.parent",),
        warnings=("test operation warning",),
        limitations=("test fixture; no scientific meaning",),
    )
    return operation, calls


def _runner(executor=None, *, implementation_version="0.4.1"):
    operation, calls = _definition(
        executor, implementation_version=implementation_version
    )
    registry = OperationRegistry()
    registry.register(operation)
    return OperationRunner(registry), operation, calls


def _prepare(runner, *, value=6, factor=3, context=None):
    return runner.prepare(
        "neuroscope.test.scale",
        "1.3.0",
        {"value": value},
        {"factor": factor},
        context=context or _context(),
    )


def test_registry_import_is_independent_of_qt_and_matplotlib():
    import os
    import subprocess
    import sys

    script = """
import sys
from neuroscope.analysis.registry import OperationRegistry
assert OperationRegistry
assert not any(name == 'PySide6' or name.startswith('PySide6.') for name in sys.modules)
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )

    assert result.returncode == 0, result.stderr

def test_operation_definition_and_registry_use_stable_ids_and_versions():
    runner, operation, _ = _runner()
    registry = OperationRegistry()
    registry.register(operation)

    assert registry.get("neuroscope.test.scale") is operation
    assert registry.get("neuroscope.test.scale", version="1.3.0") is operation
    assert registry.list_operations() == (operation,)
    assert registry.available() == [operation]
    assert AnalysisRegistry is OperationRegistry
    assert runner is not None
    assert operation.version == "1.3.0"
    assert operation.implementation_version == "0.4.1"

    with pytest.raises(ValueError, match="version mismatch"):
        registry.get("neuroscope.test.scale", version="2.0.0")


def test_registry_rejects_duplicate_ids_even_when_versions_differ():
    operation, _ = _definition()
    updated, _ = _definition(implementation_version="0.4.2")
    registry = OperationRegistry()
    registry.register(operation)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(updated)


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"operation_id": "Scale"}, ValueError),
        ({"version": "latest"}, ValueError),
        ({"implementation_version": ""}, ValueError),
        ({"input_model": dict}, TypeError),
        ({"execute": None}, TypeError),
    ],
)
def test_invalid_operation_definitions_are_rejected(changes, error):
    operation, _ = _definition()
    definition_values = {
        "operation_id": operation.operation_id,
        "name": operation.name,
        "description": operation.description,
        "version": operation.version,
        "implementation_version": operation.implementation_version,
        "input_model": operation.input_model,
        "parameters_model": operation.parameters_model,
        "output_model": operation.output_model,
        "execute": operation.execute,
        "metadata_requirements": operation.metadata_requirements,
        "prerequisites": operation.prerequisites,
        "warnings": operation.warnings,
        "limitations": operation.limitations,
    }
    definition_values.update(changes)

    with pytest.raises(error):
        OperationDefinition(**definition_values)


def test_preparation_validates_and_snapshots_without_executing():
    runner, _, calls = _runner()
    prepared = _prepare(runner)

    assert calls == []
    assert prepared.operation_id == "neuroscope.test.scale"
    assert prepared.operation_version == "1.3.0"
    assert prepared.implementation_version == "0.4.1"
    assert prepared.input_data == ScalarInput(value=6)
    assert prepared.parameters == ScaleParameters(factor=3)
    assert prepared.context.selected_channels == (1,)
    assert prepared.context.sample_interval == SampleInterval(start=20, stop=80)

    # Properties return deserialized snapshots, so mutating a returned copy does
    # not mutate the request that approval will bind to.
    copy = prepared.parameters.model_copy()
    with pytest.raises(ValidationError):
        copy.factor = 99
    assert prepared.parameters.factor == 3


def test_numpy_arrays_cannot_be_embedded_in_prepared_operation_inputs():
    calls = []

    def execute(input_data, parameters, context):
        calls.append(True)
        return ScalarOutput(value=1)

    operation = OperationDefinition(
        operation_id="neuroscope.test.reject-array",
        name="Test array rejection",
        description="Ensures data references, not arrays, enter prepared requests.",
        version="1.0.0",
        implementation_version="test-1",
        input_model=ArrayInput,
        parameters_model=ScaleParameters,
        output_model=ScalarOutput,
        execute=execute,
    )
    registry = OperationRegistry()
    registry.register(operation)
    runner = OperationRunner(registry)

    with pytest.raises(OperationValidationError, match="use DataReference"):
        runner.prepare(
            operation.operation_id,
            operation.version,
            ArrayInput(samples=np.arange(8)),
            ScaleParameters(factor=1),
        )
    assert calls == []


def test_invalid_input_or_parameters_fail_before_the_executor_runs():
    runner, _, calls = _runner()

    with pytest.raises(OperationValidationError):
        runner.prepare(
            "neuroscope.test.scale",
            "1.3.0",
            {"value": "6"},
            {"factor": 3},
            context=_context(),
        )
    with pytest.raises(OperationValidationError):
        runner.prepare(
            "neuroscope.test.scale",
            "1.3.0",
            {"value": 6},
            {"factor": 0},
            context=_context(),
        )
    assert calls == []


@pytest.mark.parametrize(
    "context, message",
    [
        (OperationContext(), "missing or unknown"),
        (_context(rate_status=MetadataStatus.INFERRED), "must be confirmed"),
        (_context(rate=None, rate_status=MetadataStatus.UNKNOWN), "missing or unknown"),
    ],
)
def test_required_metadata_must_be_present_and_confirmed(context, message):
    runner, _, calls = _runner()

    with pytest.raises(OperationValidationError, match=message):
        _prepare(runner, context=context)
    assert calls == []


def test_unknown_optional_metadata_is_preserved_without_inventing_units():
    runner, _, _ = _runner()
    prepared = _prepare(runner)
    approved = runner.approve(prepared, approved_by="researcher")
    record = runner.execute(approved)

    units = {item.name: item for item in record.input_metadata}["units"]
    output_units = {item.name: item for item in record.output_metadata}["units"]
    assert units.value is None and units.status is MetadataStatus.UNKNOWN
    assert output_units.value is None and output_units.status is MetadataStatus.UNKNOWN


def test_prerequisite_references_are_required_successful_and_declared():
    runner, _, calls = _runner()
    no_prerequisite = _context().model_copy(update={"prerequisites": ()})
    failed_prerequisite = _context().model_copy(
        update={
            "prerequisites": (
                PrerequisiteReference(
                    execution_id="failed-run",
                    operation_id="neuroscope.test.parent",
                    operation_version="1.2.0",
                    output_reference="artifact://failed",
                    status=ExecutionStatus.FAILED,
                ),
            )
        }
    )
    undeclared_prerequisite = _context().model_copy(
        update={
            "prerequisites": _context().prerequisites
            + (
                PrerequisiteReference(
                    execution_id="other-run",
                    operation_id="neuroscope.test.other",
                    operation_version="1.0.0",
                    output_reference="artifact://other",
                ),
            )
        }
    )

    for context, message in (
        (no_prerequisite, "missing prerequisite"),
        (failed_prerequisite, "must have succeeded"),
        (undeclared_prerequisite, "undeclared prerequisite"),
    ):
        with pytest.raises(OperationValidationError, match=message):
            _prepare(runner, context=context)
    assert calls == []


def test_operation_version_is_checked_during_preparation():
    runner, _, calls = _runner()
    with pytest.raises(ValueError, match="version mismatch"):
        runner.prepare(
            "neuroscope.test.scale",
            "1.2.0",
            {"value": 1},
            {"factor": 2},
            context=_context(),
        )
    assert calls == []


def test_unapproved_requests_cannot_execute_and_approval_is_explicit():
    runner, _, calls = _runner()
    prepared = _prepare(runner)

    with pytest.raises(ApprovalRequiredError):
        runner.execute(prepared)
    with pytest.raises(ValueError, match="approved_by"):
        runner.approve(prepared, approved_by=" ")
    assert calls == []

    approval = runner.approve(prepared, approved_by="researcher-17")
    assert approval.authorizes(prepared)


def test_approval_fingerprint_changes_with_params_input_metadata_or_prerequisites():
    runner, _, _ = _runner()
    original = _prepare(runner)
    approval = runner.approve(original, approved_by="researcher")
    changed_requests = (
        _prepare(runner, factor=4),
        _prepare(runner, value=7),
        _prepare(runner, context=_context(dataset_id="dataset-b")),
        _prepare(runner, context=_context(rate=2000.0)),
        _prepare(
            runner,
            context=_context(prerequisite_execution="parent-run-b"),
        ),
    )

    for changed in changed_requests:
        assert not approval.authorizes(changed)
        assert changed.request_fingerprint != original.request_fingerprint


def test_execution_returns_typed_success_record_and_preserves_provenance():
    runner, _, calls = _runner()
    prepared = _prepare(runner)
    approval = runner.approve(prepared, approved_by="researcher-17")

    first = runner.execute(approval)
    second = runner.execute(approval)

    assert calls == [(6, 3), (6, 3)]
    assert first.status is ExecutionStatus.SUCCEEDED
    assert first.error is None
    assert first.output == ScalarOutput(value=18)
    assert first.parameters == ScaleParameters(factor=3)
    assert first.operation_version == "1.3.0"
    assert first.implementation_version == "0.4.1"
    assert first.provenance.approved_by == "researcher-17"
    assert first.provenance.approval_id == approval.approval_id
    assert first.provenance.dataset_reference.dataset_id == "dataset-a"
    assert first.input_content_fingerprint == "sha256:abc123"
    assert first.selected_channels == (1,)
    assert first.sample_interval == SampleInterval(start=20, stop=80)
    assert first.prerequisite_references[0].execution_id == "parent-run-a"
    assert first.reproducibility.seed == 31
    assert first.warnings == ("test operation warning", "test execution warning")
    assert first.started_at_utc.tzinfo is timezone.utc
    assert first.finished_at_utc.tzinfo is timezone.utc
    assert first.execution_id != second.execution_id
    assert first.output == second.output


def test_execution_failure_is_returned_as_a_structured_record():
    def fail(_input, _parameters, _context):
        raise RuntimeError("deterministic test failure")

    runner, _, calls = _runner(executor=fail)
    prepared = _prepare(runner)
    record = runner.execute(runner.approve(prepared, approved_by="researcher"))

    assert calls == [(6, 3)]
    assert record.status is ExecutionStatus.FAILED
    assert record.output is None
    assert record.error.error_type == "RuntimeError"
    assert record.error.message == "deterministic test failure"
    assert record.parameters.factor == 3


def test_numpy_arrays_are_not_serialized_into_execution_records():
    def array_output(_input, _parameters, _context):
        return ArrayOutput(samples=np.arange(16))

    operation = OperationDefinition(
        operation_id="neuroscope.test.array-output",
        name="Test array output rejection",
        description="Array results require a reference-backed artifact store.",
        version="1.0.0",
        implementation_version="test-1",
        input_model=ScalarInput,
        parameters_model=ScaleParameters,
        output_model=ArrayOutput,
        execute=array_output,
    )
    registry = OperationRegistry()
    registry.register(operation)
    runner = OperationRunner(registry)
    prepared = runner.prepare(
        operation.operation_id,
        operation.version,
        ScalarInput(value=1),
        ScaleParameters(factor=1),
    )
    record = runner.execute(runner.approve(prepared, approved_by="researcher"))

    assert record.status is ExecutionStatus.FAILED
    assert record.output is None
    assert record.error.error_type == "OperationValidationError"
    assert "DataReference" in record.error.message
    assert record._output_json is None


def test_output_schema_failure_is_captured_as_execution_failure():
    def invalid_output(_input, _parameters, _context):
        return {"value": "not-an-integer"}

    runner, _, _ = _runner(executor=invalid_output)
    prepared = _prepare(runner)
    record = runner.execute(runner.approve(prepared, approved_by="researcher"))

    assert record.status is ExecutionStatus.FAILED
    assert record.output is None
    assert record.error.error_type == "ValidationError"


def test_approval_cannot_be_used_by_another_runner():
    runner_a, _, _ = _runner()
    prepared = _prepare(runner_a)
    approval = runner_a.approve(prepared, approved_by="researcher")
    runner_b, _, calls_b = _runner()

    with pytest.raises(ApprovalMismatchError, match="this runner"):
        runner_b.execute(approval)
    assert calls_b == []


def test_original_array_is_not_embedded_or_modified_by_generic_execution():
    source = np.arange(20, dtype=np.float64)
    before = source.copy()
    runner, _, _ = _runner()
    prepared = _prepare(runner)
    record = runner.execute(runner.approve(prepared, approved_by="researcher"))

    np.testing.assert_array_equal(source, before)
    assert "source" not in repr(record.input_data)

