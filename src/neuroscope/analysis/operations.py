"""Synchronous prepare/approve/execute lifecycle for scientific operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from inspect import signature
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ValidationError

from ..artifacts import ISIArtifactStore, WaveformArtifactStore
from ..data_access import DatasetAccessError, DatasetStore, ResolvedSignalData
from ..exploration import MetadataStatus
from ..mua_artifacts import MUAArtifactStore
from .contracts import (
    DataReference,
    DatasetReference,
    ExecutionStatus,
    MetadataSnapshot,
    OperationContext,
    OperationDefinition,
    OperationModel,
    OperationResult,
    PrerequisiteReference,
    ReproducibilityInfo,
    SampleInterval,
)
from .registry import OperationRegistry


class OperationValidationError(ValueError):
    """Input, parameter, metadata, or prerequisite validation failed."""


class ApprovalRequiredError(PermissionError):
    """Execution was requested without a runner-issued approval object."""


class ApprovalMismatchError(PermissionError):
    """Approval does not authorize this exact prepared request or implementation."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _validate_model(model_type: type[OperationModel], value: Any) -> OperationModel:
    """Revalidate model instances too, including any unchecked model_copy updates."""
    payload = value.model_dump(mode="python") if isinstance(value, BaseModel) else value
    return model_type.model_validate(payload, strict=True)


def _reject_embedded_arrays(value: Any) -> None:
    """Prevent signal arrays from entering serialized requests or provenance."""
    if isinstance(value, np.ndarray):
        raise OperationValidationError(
            "NumPy arrays cannot be embedded in operation requests; use DataReference."
        )
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            _reject_embedded_arrays(getattr(value, field_name))
    elif isinstance(value, Mapping):
        for nested in value.values():
            _reject_embedded_arrays(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            _reject_embedded_arrays(nested)


def _request_fingerprint(
    definition: OperationDefinition,
    input_json: str,
    parameters_json: str,
    context_json: str,
) -> str:
    payload = {
        "operation_id": definition.operation_id,
        "operation_version": definition.version,
        "implementation_version": definition.implementation_version,
        "input": json.loads(input_json),
        "parameters": json.loads(parameters_json),
        "context": json.loads(context_json),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedExecution:
    """Effectively immutable normalized request; property access returns fresh models."""

    operation_id: str
    operation_version: str
    implementation_version: str
    request_fingerprint: str
    _definition: OperationDefinition
    _input_json: str
    _parameters_json: str
    _context_json: str
    warnings: tuple[str, ...] = ()

    @property
    def input_data(self) -> OperationModel:
        return self._definition.input_model.model_validate_json(
            self._input_json, strict=True
        )

    @property
    def parameters(self) -> OperationModel:
        return self._definition.parameters_model.model_validate_json(
            self._parameters_json, strict=True
        )

    @property
    def context(self) -> OperationContext:
        return OperationContext.model_validate_json(self._context_json, strict=True)

    def authorizes(self, prepared: PreparedExecution) -> bool:
        """True only for a request with the same normalized execution fingerprint."""
        return (
            isinstance(prepared, PreparedExecution)
            and self.request_fingerprint == prepared.request_fingerprint
        )

    def to_snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe request/provenance snapshot without implementation objects."""
        return {
            "operation_id": self.operation_id,
            "operation_version": self.operation_version,
            "implementation_version": self.implementation_version,
            "request_fingerprint": self.request_fingerprint,
            "warnings": list(self.warnings),
            "input": json.loads(self._input_json),
            "parameters": json.loads(self._parameters_json),
            "context": json.loads(self._context_json),
        }


_APPROVAL_SEAL = object()


@dataclass(frozen=True, slots=True, init=False)
class ApprovedExecution:
    """Runner-issued approval bound to one immutable prepared request."""

    prepared: PreparedExecution
    approval_id: str
    approved_by: str
    approved_at_utc: datetime
    _runner_token: object
    _seal: object

    def __init__(
        self,
        prepared: PreparedExecution,
        approved_by: str,
        approved_at_utc: datetime,
        runner_token: object,
        *,
        _seal: object,
    ) -> None:
        if _seal is not _APPROVAL_SEAL:
            raise TypeError("ApprovedExecution values must be created by OperationRunner.approve().")
        object.__setattr__(self, "prepared", prepared)
        object.__setattr__(self, "approval_id", str(uuid4()))
        object.__setattr__(self, "approved_by", approved_by)
        object.__setattr__(self, "approved_at_utc", approved_at_utc)
        object.__setattr__(self, "_runner_token", runner_token)
        object.__setattr__(self, "_seal", _seal)

    def authorizes(self, prepared: PreparedExecution) -> bool:
        return self.prepared.authorizes(prepared)

    def to_snapshot(self) -> dict[str, Any]:
        """Return serializable approval evidence; this cannot recreate authority."""
        return {
            "approval_id": self.approval_id,
            "approved_by": self.approved_by,
            "approved_at_utc": self.approved_at_utc.isoformat(),
            "prepared": self.prepared.to_snapshot(),
        }


@dataclass(frozen=True, slots=True)
class OperationFailure:
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class ExecutionProvenance:
    dataset_reference: DatasetReference | None
    prerequisite_references: tuple[PrerequisiteReference, ...]
    approval_id: str
    approved_by: str
    approved_at_utc: datetime
    reproducibility: ReproducibilityInfo | None


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """In-memory typed result and provenance snapshot for one attempted execution."""

    execution_id: str
    operation_id: str
    operation_version: str
    implementation_version: str
    status: ExecutionStatus
    started_at_utc: datetime
    finished_at_utc: datetime
    provenance: ExecutionProvenance
    warnings: tuple[str, ...]
    error: OperationFailure | None
    _input_json: str
    _parameters_json: str
    _context_json: str
    _output_json: str | None
    _output_metadata_json: str
    _definition: OperationDefinition

    @property
    def data_references(self) -> tuple[DataReference, ...]:
        return _collect_data_references(self.input_data)

    @property
    def input_reference(self) -> DatasetReference | None:
        return self.context.dataset_reference

    @property
    def input_content_fingerprint(self) -> str | None:
        reference = self.input_reference
        return reference.content_fingerprint if reference is not None else None

    @property
    def input_metadata(self) -> tuple[MetadataSnapshot, ...]:
        return self.context.metadata

    @property
    def selected_channels(self) -> tuple[int, ...]:
        return self.context.selected_channels

    @property
    def sample_interval(self) -> SampleInterval | None:
        return self.context.sample_interval

    @property
    def prerequisite_references(self) -> tuple[PrerequisiteReference, ...]:
        return self.context.prerequisites

    @property
    def reproducibility(self) -> ReproducibilityInfo | None:
        return self.context.reproducibility

    @property
    def input_data(self) -> OperationModel:
        return self._definition.input_model.model_validate_json(
            self._input_json, strict=True
        )

    @property
    def parameters(self) -> OperationModel:
        return self._definition.parameters_model.model_validate_json(
            self._parameters_json, strict=True
        )

    @property
    def context(self) -> OperationContext:
        return OperationContext.model_validate_json(self._context_json, strict=True)

    @property
    def output(self) -> OperationModel | None:
        if self._output_json is None:
            return None
        return self._definition.output_model.model_validate_json(
            self._output_json, strict=True
        )

    @property
    def output_metadata(self) -> tuple[MetadataSnapshot, ...]:
        raw = json.loads(self._output_metadata_json)
        return tuple(MetadataSnapshot.model_validate(item) for item in raw)

    def to_snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe record including approval and typed scientific outputs."""
        return {
            "execution_id": self.execution_id,
            "operation_id": self.operation_id,
            "operation_version": self.operation_version,
            "implementation_version": self.implementation_version,
            "status": self.status.value,
            "started_at_utc": self.started_at_utc.isoformat(),
            "finished_at_utc": self.finished_at_utc.isoformat(),
            "input_data": self.input_data.model_dump(mode="json"),
            "parameters": self.parameters.model_dump(mode="json"),
            "context": self.context.model_dump(mode="json"),
            "output": (
                self.output.model_dump(mode="json")
                if self.output is not None
                else None
            ),
            "output_metadata": [item.model_dump(mode="json") for item in self.output_metadata],
            "warnings": list(self.warnings),
            "error": (
                {"error_type": self.error.error_type, "message": self.error.message}
                if self.error is not None
                else None
            ),
            "provenance": {
                "dataset_reference": (
                    self.provenance.dataset_reference.model_dump(mode="json")
                    if self.provenance.dataset_reference is not None
                    else None
                ),
                "prerequisite_references": [
                    item.model_dump(mode="json")
                    for item in self.provenance.prerequisite_references
                ],
                "approval_id": self.provenance.approval_id,
                "approved_by": self.provenance.approved_by,
                "approved_at_utc": self.provenance.approved_at_utc.isoformat(),
                "reproducibility": (
                    self.provenance.reproducibility.model_dump(mode="json")
                    if self.provenance.reproducibility is not None
                    else None
                ),
            },
        }


def restore_execution_record(
    snapshot: Mapping[str, Any], definition: OperationDefinition
) -> ExecutionRecord:
    """Validate and restore a historical record without restoring approval authority.

    The operation callable is never invoked. Operation and implementation versions
    must exactly match the registered schema so incompatible history remains visible
    as unsupported instead of being reinterpreted.
    """
    if snapshot.get("operation_id") != definition.operation_id:
        raise ValueError("execution snapshot operation ID does not match its definition.")
    if snapshot.get("operation_version") != definition.version:
        raise ValueError("execution snapshot operation version is unsupported.")
    if snapshot.get("implementation_version") != definition.implementation_version:
        raise ValueError("execution snapshot implementation version is unsupported.")
    required = {
        "execution_id", "status", "started_at_utc", "finished_at_utc",
        "input_data", "parameters", "context", "output_metadata", "warnings",
        "provenance",
    }
    if not required.issubset(snapshot):
        raise ValueError("execution snapshot is missing required fields.")
    input_data = definition.input_model.model_validate_json(
        _canonical_json(snapshot["input_data"]), strict=True
    )
    parameters = definition.parameters_model.model_validate_json(
        _canonical_json(snapshot["parameters"]), strict=True
    )
    context = OperationContext.model_validate_json(
        _canonical_json(snapshot["context"]), strict=True
    )
    _reject_embedded_arrays(input_data)
    _reject_embedded_arrays(parameters)
    provenance_data = snapshot["provenance"]
    if not isinstance(provenance_data, Mapping):
        raise TypeError("execution snapshot provenance must be an object.")
    dataset_value = provenance_data.get("dataset_reference")
    dataset_reference = (
        DatasetReference.model_validate_json(_canonical_json(dataset_value), strict=True)
        if dataset_value is not None else None
    )
    prerequisites = tuple(
        PrerequisiteReference.model_validate_json(_canonical_json(item), strict=True)
        for item in provenance_data.get("prerequisite_references", ())
    )
    reproducibility_value = provenance_data.get("reproducibility")
    reproducibility = (
        ReproducibilityInfo.model_validate_json(
            _canonical_json(reproducibility_value), strict=True
        )
        if reproducibility_value is not None else None
    )
    if dataset_reference != context.dataset_reference or prerequisites != context.prerequisites:
        raise ValueError("execution snapshot provenance does not match its context.")
    if reproducibility != context.reproducibility:
        raise ValueError("execution snapshot reproducibility provenance does not match its context.")
    output_json = None
    if snapshot.get("output") is not None:
        output = definition.output_model.model_validate_json(
            _canonical_json(snapshot["output"]), strict=True
        )
        _reject_embedded_arrays(output)
        output_json = output.model_dump_json()
    output_metadata = tuple(
        MetadataSnapshot.model_validate_json(_canonical_json(item), strict=True)
        for item in snapshot["output_metadata"]
    )
    error_data = snapshot.get("error")
    failure = (
        OperationFailure(str(error_data["error_type"]), str(error_data["message"]))
        if isinstance(error_data, Mapping) else None
    )
    status = ExecutionStatus(snapshot["status"])
    if (status is ExecutionStatus.SUCCEEDED) != (output_json is not None and failure is None):
        raise ValueError("execution snapshot status conflicts with its output or error.")
    if status is ExecutionStatus.FAILED and failure is None:
        raise ValueError("failed execution snapshot is missing its error information.")
    approval_id = str(provenance_data["approval_id"])
    approved_by = str(provenance_data["approved_by"])
    approved_at = datetime.fromisoformat(str(provenance_data["approved_at_utc"]))
    if approved_at.tzinfo is None or approved_at.utcoffset() is None:
        raise ValueError("execution approval timestamp must include a timezone.")
    started = datetime.fromisoformat(str(snapshot["started_at_utc"]))
    finished = datetime.fromisoformat(str(snapshot["finished_at_utc"]))
    if started.tzinfo is None or finished.tzinfo is None or finished < started:
        raise ValueError("execution timestamps must be timezone-aware and ordered.")
    warnings = snapshot["warnings"]
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise ValueError("execution warnings must be a list of strings.")
    return ExecutionRecord(
        execution_id=str(snapshot["execution_id"]),
        operation_id=definition.operation_id,
        operation_version=definition.version,
        implementation_version=definition.implementation_version,
        status=status,
        started_at_utc=started,
        finished_at_utc=finished,
        provenance=ExecutionProvenance(
            dataset_reference=dataset_reference,
            prerequisite_references=prerequisites,
            approval_id=approval_id,
            approved_by=approved_by,
            approved_at_utc=approved_at,
            reproducibility=reproducibility,
        ),
        warnings=tuple(warnings),
        error=failure,
        _input_json=_canonical_json(input_data.model_dump(mode="json")),
        _parameters_json=_canonical_json(parameters.model_dump(mode="json")),
        _context_json=_canonical_json(context.model_dump(mode="json")),
        _output_json=output_json,
        _output_metadata_json=_canonical_json(
            [item.model_dump(mode="json") for item in output_metadata]
        ),
        _definition=definition,
    )


def _collect_data_references(value: Any) -> tuple[DataReference, ...]:
    """Collect only typed references from a validated operation input model."""
    found: list[DataReference] = []

    def visit(item: Any) -> None:
        if isinstance(item, DataReference):
            found.append(item)
        elif isinstance(item, BaseModel):
            for field_name in type(item).model_fields:
                visit(getattr(item, field_name))
        elif isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (tuple, list)):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(found)


@dataclass(frozen=True, slots=True)
class OperationRuntimeContext:
    """Execution-only context containing only the data named by input references."""

    request_context: OperationContext
    resolved_data: tuple[ResolvedSignalData, ...]
    artifact_store: WaveformArtifactStore | None
    isi_artifact_store: ISIArtifactStore | None
    mua_artifact_store: MUAArtifactStore | None
    execution_id: str
    started_at_utc: datetime
    approval_id: str = ""
    approved_by: str = ""

    def data_for(self, reference: DataReference) -> ResolvedSignalData:
        for resolved in self.resolved_data:
            if resolved.reference == reference:
                return resolved
        raise KeyError("the operation did not declare this data reference in its input.")


class OperationRunner:
    """Validates, obtains approval, and synchronously runs registered operations."""

    def __init__(
        self,
        registry: OperationRegistry,
        *,
        data_store: DatasetStore | None = None,
        artifact_store: WaveformArtifactStore | None = None,
        isi_artifact_store: ISIArtifactStore | None = None,
        mua_artifact_store: MUAArtifactStore | None = None,
    ) -> None:
        if not isinstance(registry, OperationRegistry):
            raise TypeError("registry must be an OperationRegistry.")
        if data_store is not None and not isinstance(data_store, DatasetStore):
            raise TypeError("data_store must be a DatasetStore or None.")
        if artifact_store is not None and not isinstance(
            artifact_store, WaveformArtifactStore
        ):
            raise TypeError("artifact_store must be a WaveformArtifactStore or None.")
        if isi_artifact_store is not None and not isinstance(
            isi_artifact_store, ISIArtifactStore
        ):
            raise TypeError("isi_artifact_store must be an ISIArtifactStore or None.")
        if mua_artifact_store is not None and not isinstance(
            mua_artifact_store, MUAArtifactStore
        ):
            raise TypeError("mua_artifact_store must be an MUAArtifactStore or None.")
        self._registry = registry
        self._data_store = data_store
        self._artifact_store = artifact_store
        self._isi_artifact_store = isi_artifact_store
        self._mua_artifact_store = mua_artifact_store
        self._runner_token = object()

    def prepare(
        self,
        operation_id: str,
        operation_version: str,
        input_data: OperationModel | dict[str, Any],
        parameters: OperationModel | dict[str, Any],
        *,
        context: OperationContext | dict[str, Any] | None = None,
    ) -> PreparedExecution:
        """Validate and snapshot a request without calling its implementation."""
        definition = self._registry.get(operation_id, version=operation_version)
        try:
            normalized_input = _validate_model(definition.input_model, input_data)
            normalized_parameters = _validate_model(
                definition.parameters_model, parameters
            )
            normalized_context = _validate_model(
                OperationContext, {} if context is None else context
            )
        except ValidationError as exc:
            raise OperationValidationError(str(exc)) from exc

        self._validate_context(definition, normalized_context)
        _reject_embedded_arrays(normalized_input)
        _reject_embedded_arrays(normalized_parameters)
        _reject_embedded_arrays(normalized_context)
        try:
            input_json = normalized_input.model_dump_json()
            parameters_json = normalized_parameters.model_dump_json()
            context_json = normalized_context.model_dump_json()
            # Require every input snapshot to be canonically serializable.
            json.loads(input_json)
            json.loads(parameters_json)
            json.loads(context_json)
        except (TypeError, ValueError) as exc:
            raise OperationValidationError(
                f"operation request must have JSON-serializable snapshots: {exc}"
            ) from exc

        preparation_warnings = (
            definition.prepare_warnings(
                normalized_input, normalized_parameters, normalized_context
            )
            if definition.prepare_warnings is not None
            else ()
        )
        if not isinstance(preparation_warnings, tuple) or any(
            not isinstance(item, str) or not item for item in preparation_warnings
        ):
            raise OperationValidationError(
                "prepare_warnings must return a tuple of non-empty strings."
            )

        fingerprint = _request_fingerprint(
            definition, input_json, parameters_json, context_json
        )
        return PreparedExecution(
            operation_id=definition.operation_id,
            operation_version=definition.version,
            implementation_version=definition.implementation_version,
            request_fingerprint=fingerprint,
            warnings=preparation_warnings,
            _definition=definition,
            _input_json=input_json,
            _parameters_json=parameters_json,
            _context_json=context_json,
        )

    def approve(
        self,
        prepared: PreparedExecution,
        *,
        approved_by: str,
    ) -> ApprovedExecution:
        """Record an explicit approval for the exact normalized request."""
        if not isinstance(prepared, PreparedExecution):
            raise TypeError("approve() requires a PreparedExecution.")
        if not isinstance(approved_by, str) or not approved_by.strip():
            raise ValueError("approved_by must identify the approving researcher.")
        self._assert_registered_version(prepared)
        expected = _request_fingerprint(
            prepared._definition,
            prepared._input_json,
            prepared._parameters_json,
            prepared._context_json,
        )
        if expected != prepared.request_fingerprint:
            raise ApprovalMismatchError("prepared request snapshot has changed.")
        return ApprovedExecution(
            prepared,
            approved_by.strip(),
            datetime.now(timezone.utc),
            self._runner_token,
            _seal=_APPROVAL_SEAL,
        )

    def execute(self, approved: ApprovedExecution) -> ExecutionRecord:
        """Execute only a runner-issued approval and return success or failure data."""
        if not isinstance(approved, ApprovedExecution):
            raise ApprovalRequiredError(
                "execution requires a runner-issued ApprovedExecution value."
            )
        if approved._seal is not _APPROVAL_SEAL or approved._runner_token is not self._runner_token:
            raise ApprovalMismatchError("approval was not issued by this runner.")
        prepared = approved.prepared
        if not approved.authorizes(prepared):
            raise ApprovalMismatchError("approval does not match the prepared request.")
        self._assert_registered_version(prepared)
        expected = _request_fingerprint(
            prepared._definition,
            prepared._input_json,
            prepared._parameters_json,
            prepared._context_json,
        )
        if expected != prepared.request_fingerprint:
            raise ApprovalMismatchError("approved request snapshot has changed.")

        definition = prepared._definition
        try:
            input_model = definition.input_model.model_validate_json(
                prepared._input_json, strict=True
            )
            parameter_model = definition.parameters_model.model_validate_json(
                prepared._parameters_json, strict=True
            )
            context = OperationContext.model_validate_json(
                prepared._context_json, strict=True
            )
            self._validate_context(definition, context)
        except (ValidationError, OperationValidationError) as exc:
            raise ApprovalMismatchError(
                f"approved request no longer validates: {exc}"
            ) from exc

        started = datetime.now(timezone.utc)
        execution_id = str(uuid4())
        try:
            references = _collect_data_references(input_model)
            try:
                signature(definition.execute).bind(
                    input_model, parameter_model, context, None
                )
                accepts_runtime_access = True
            except TypeError:
                accepts_runtime_access = False
            if references and not accepts_runtime_access:
                raise DatasetAccessError(
                    "operation implementation must accept runtime access to consume DataReference inputs."
                )
            if references and self._data_store is None:
                raise DatasetAccessError(
                    "operation input contains data references, but no DatasetStore is configured."
                )
            resolved_data = (
                tuple(self._data_store.resolve(reference) for reference in references)
                if references and self._data_store is not None
                else ()
            )
            runtime_context = OperationRuntimeContext(
                context,
                resolved_data,
                self._artifact_store,
                self._isi_artifact_store,
                self._mua_artifact_store,
                execution_id,
                started,
                approved.approval_id,
                approved.approved_by,
            )
            if accepts_runtime_access:
                raw_result = definition.execute(
                    input_model, parameter_model, context, runtime_context
                )
            else:
                raw_result = definition.execute(input_model, parameter_model, context)
            if isinstance(raw_result, OperationResult):
                output_candidate = raw_result.output
                output_metadata = raw_result.output_metadata
                dynamic_warnings = raw_result.warnings
            else:
                output_candidate = raw_result
                output_metadata = ()
                dynamic_warnings = ()
            output = definition.output_model.model_validate(
                output_candidate, strict=True
            )
            _reject_embedded_arrays(output)
            self._validate_output_metadata(output_metadata)
            output_json = output.model_dump_json()
            output_metadata_json = json.dumps(
                [item.model_dump(mode="json") for item in output_metadata],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            warnings = tuple(dict.fromkeys(
                definition.warnings + prepared.warnings + tuple(dynamic_warnings)
            ))
            failure = None
            status = ExecutionStatus.SUCCEEDED
        except Exception as exc:  # noqa: BLE001 - operation failures become typed records.
            output_json = None
            output_metadata_json = "[]"
            warnings = tuple(dict.fromkeys(definition.warnings + prepared.warnings))
            failure = OperationFailure(type(exc).__name__, str(exc))
            status = ExecutionStatus.FAILED

        finished = datetime.now(timezone.utc)
        return ExecutionRecord(
            execution_id=execution_id,
            operation_id=definition.operation_id,
            operation_version=definition.version,
            implementation_version=definition.implementation_version,
            status=status,
            started_at_utc=started,
            finished_at_utc=finished,
            provenance=ExecutionProvenance(
                dataset_reference=context.dataset_reference,
                prerequisite_references=context.prerequisites,
                approval_id=approved.approval_id,
                approved_by=approved.approved_by,
                approved_at_utc=approved.approved_at_utc,
                reproducibility=context.reproducibility,
            ),
            warnings=warnings,
            error=failure,
            _input_json=prepared._input_json,
            _parameters_json=prepared._parameters_json,
            _context_json=prepared._context_json,
            _output_json=output_json,
            _output_metadata_json=output_metadata_json,
            _definition=definition,
        )

    @staticmethod
    def _validate_context(
        definition: OperationDefinition,
        context: OperationContext,
    ) -> None:
        metadata = {item.name: item for item in context.metadata}
        for requirement in definition.metadata_requirements:
            item = metadata.get(requirement.name)
            if not requirement.required:
                continue
            if item is None or item.value is None or item.status is MetadataStatus.UNKNOWN:
                raise OperationValidationError(
                    f"required metadata {requirement.name!r} is missing or unknown."
                )
            if requirement.require_confirmed and item.status is not MetadataStatus.CONFIRMED:
                raise OperationValidationError(
                    f"metadata {requirement.name!r} must be confirmed."
                )

        required_prerequisites = set(definition.prerequisites)
        provided_prerequisites = {
            item.operation_id for item in context.prerequisites
        }
        missing = required_prerequisites - provided_prerequisites
        unexpected = provided_prerequisites - required_prerequisites
        if missing:
            raise OperationValidationError(
                f"missing prerequisite operation references: {', '.join(sorted(missing))}."
            )
        if unexpected:
            raise OperationValidationError(
                f"undeclared prerequisite operation references: {', '.join(sorted(unexpected))}."
            )
        if any(item.status != ExecutionStatus.SUCCEEDED for item in context.prerequisites):
            raise OperationValidationError("all prerequisite executions must have succeeded.")

    @staticmethod
    def _validate_output_metadata(metadata: tuple[MetadataSnapshot, ...]) -> None:
        if any(not isinstance(item, MetadataSnapshot) for item in metadata):
            raise TypeError("operation output metadata must use MetadataSnapshot values.")
        names = [item.name for item in metadata]
        if len(set(names)) != len(names):
            raise ValueError("output metadata names must be unique.")

    def _assert_registered_version(self, prepared: PreparedExecution) -> None:
        registered = self._registry.get(prepared.operation_id)
        definition = prepared._definition
        if (
            registered is not definition
            or registered.version != prepared.operation_version
            or registered.implementation_version != prepared.implementation_version
        ):
            raise ApprovalMismatchError(
                "operation or implementation version changed after preparation."
            )
