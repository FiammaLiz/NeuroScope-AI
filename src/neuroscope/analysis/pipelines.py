"""Versioned, approval-bound scientific pipeline definitions and execution.

Pipeline records contain durable identities and bindings only. Session handles,
resolved arrays, and runner approvals are created at runtime and are never saved.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Literal, get_args, get_origin
from uuid import uuid4

from pydantic import (
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from ..data_access import DatasetAccessError, DatasetStore
from .contracts import (
    DatasetReference,
    ExecutionStatus,
    MetadataSnapshot,
    OperationContext,
    OperationModel,
    PrerequisiteReference,
    SampleInterval,
)
from .isi_operation import OPERATION_ID as ISI_ID
from .isi_operation import isi_collection_from_detection_collection
from .mua_average_operation import OPERATION_ID as MUA_AVERAGE_ID
from .mua_average_operation import mua_artifact_reference_from_record
from .mua_operation import OPERATION_ID as CALC_MUA_ID
from .operations import (
    ApprovalRequiredError,
    ApprovedExecution,
    ExecutionRecord,
    OperationRunner,
    PreparedExecution,
)
from .registry import OperationRegistry
from .signed_peak_operation import OPERATION_ID as SIGNED_DETECTOR_ID
from .spike_operation import OPERATION_ID as NEGATIVE_DETECTOR_ID
from .waveform_operation import OPERATION_ID as WAVEFORM_ID
from .waveform_operation import detection_collection_from_record

PIPELINE_SCHEMA_VERSION = "1.0.0"
_DETECTORS = {NEGATIVE_DETECTOR_ID, SIGNED_DETECTOR_ID}
_SUPPORTED_EDGE_TYPES = {
    (NEGATIVE_DETECTOR_ID, WAVEFORM_ID, "detections", "detection_events"),
    (SIGNED_DETECTOR_ID, WAVEFORM_ID, "detections", "detection_events"),
    (NEGATIVE_DETECTOR_ID, ISI_ID, "events", "detection_events"),
    (SIGNED_DETECTOR_ID, ISI_ID, "events", "detection_events"),
    (CALC_MUA_ID, MUA_AVERAGE_ID, "traces", "mua_trace"),
}


class PipelineModel(OperationModel):
    """Pipeline records are frozen and reject undeclared fields."""


class PipelineInputBinding(PipelineModel):
    """Durable input binding; dataset bindings deliberately omit session handles."""

    target_field: str = Field(min_length=1)
    source_kind: Literal["dataset", "node_output", "artifact"]
    dataset: DatasetReference | None = None
    channels: tuple[int, ...] = ()
    sample_interval: SampleInterval | None = None
    component_id: str | None = None
    source_node_id: str | None = None
    source_port: Literal["detection_events", "mua_trace"] | None = None
    artifact_value: JsonValue = None

    @model_validator(mode="after")
    def source_fields_match_kind(self) -> PipelineInputBinding:
        if self.source_kind == "dataset":
            if self.dataset is None or self.sample_interval is None or not self.channels:
                raise ValueError("dataset binding requires dataset identity, channels and sample_interval.")
            if self.source_node_id or self.source_port or self.artifact_value is not None:
                raise ValueError("dataset binding contains fields for another source kind.")
            if any(index < 0 for index in self.channels) or len(set(self.channels)) != len(self.channels):
                raise ValueError("dataset binding channels must be unique non-negative indices.")
        elif self.source_kind == "node_output":
            if not self.source_node_id or not self.source_port:
                raise ValueError("node_output binding requires source_node_id and source_port.")
            if self.dataset is not None or self.sample_interval is not None or self.channels or self.artifact_value is not None:
                raise ValueError("node_output binding contains fields for another source kind.")
        else:
            if self.artifact_value is None:
                raise ValueError("artifact binding requires an explicit serialized artifact reference.")
            if self.dataset is not None or self.sample_interval is not None or self.channels or self.source_node_id or self.source_port:
                raise ValueError("artifact binding contains fields for another source kind.")
        return self


class PipelineNode(PipelineModel):
    node_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    operation_id: str = Field(min_length=1)
    operation_version: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    inputs: tuple[PipelineInputBinding, ...] = ()
    depends_on: tuple[str, ...] = ()
    reuse_execution_id: str | None = None

    @model_validator(mode="after")
    def unique_bindings(self) -> PipelineNode:
        targets = [binding.target_field for binding in self.inputs]
        duplicate_targets = {name for name in targets if targets.count(name) > 1}
        if duplicate_targets - {"traces"}:
            raise ValueError("only the MUA-traces input accepts multiple bindings.")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("node dependencies must be unique.")
        if self.node_id in self.depends_on:
            raise ValueError("a pipeline node cannot depend on itself.")
        if self.reuse_execution_id and (self.inputs or self.depends_on):
            raise ValueError("a reused execution cannot also declare inputs or dependencies.")
        return self


class PipelineDefinition(PipelineModel):
    pipeline_id: str = Field(min_length=1)
    schema_version: Literal["1.0.0"] = PIPELINE_SCHEMA_VERSION
    name: str = Field(min_length=1)
    description: str = ""
    revision: int = Field(ge=1, strict=True)
    created_at_utc: datetime
    updated_at_utc: datetime
    nodes: tuple[PipelineNode, ...] = ()

    @field_validator("created_at_utc", "updated_at_utc")
    @classmethod
    def require_utc_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("pipeline timestamps must include a timezone.")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def unique_nodes(self) -> PipelineDefinition:
        ids = [node.node_id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("pipeline node IDs must be unique.")
        return self


class PipelineNodeValidation(PipelineModel):
    node_id: str
    status: Literal["ready", "blocked", "reused", "invalid"]
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class PipelineValidation(PipelineModel):
    pipeline_id: str
    revision: int
    structurally_valid: bool
    execution_ready: bool
    topological_order: tuple[str, ...]
    nodes: tuple[PipelineNodeValidation, ...]
    errors: tuple[str, ...] = ()


class PipelineRunState(PipelineModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: str
    pipeline_id: str
    pipeline_revision: int
    started_at_utc: datetime
    updated_at_utc: datetime
    node_status: dict[str, Literal["pending", "prepared", "succeeded", "reused", "failed", "blocked"]]
    execution_ids: dict[str, str] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)


class PreparedPipelineNode(PipelineModel):
    """Serializable review surface; the live PreparedExecution remains separate."""

    run_id: str
    node_id: str
    operation_id: str
    operation_version: str
    implementation_version: str
    request_snapshot: dict[str, JsonValue]
    warnings: tuple[str, ...] = ()


def new_pipeline(
    name: str,
    nodes: tuple[PipelineNode, ...],
    *,
    description: str = "",
    pipeline_id: str | None = None,
    revision: int = 1,
    created_at_utc: datetime | None = None,
) -> PipelineDefinition:
    now = datetime.now(timezone.utc)
    return PipelineDefinition(
        pipeline_id=pipeline_id or str(uuid4()),
        name=name,
        description=description,
        revision=revision,
        created_at_utc=created_at_utc or now,
        updated_at_utc=now,
        nodes=nodes,
    )


def revise_pipeline(
    pipeline: PipelineDefinition,
    *,
    name: str | None = None,
    description: str | None = None,
    nodes: tuple[PipelineNode, ...] | None = None,
) -> PipelineDefinition:
    """Create an immutable next revision while preserving pipeline identity."""
    return PipelineDefinition(
        pipeline_id=pipeline.pipeline_id,
        schema_version=pipeline.schema_version,
        name=pipeline.name if name is None else name,
        description=pipeline.description if description is None else description,
        revision=pipeline.revision + 1,
        created_at_utc=pipeline.created_at_utc,
        updated_at_utc=datetime.now(timezone.utc),
        nodes=pipeline.nodes if nodes is None else nodes,
    )


def _topological_order(nodes: tuple[PipelineNode, ...]) -> tuple[str, ...]:
    by_id = {node.node_id: node for node in nodes}
    indegree = {node.node_id: 0 for node in nodes}
    children: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        for dependency in node.depends_on:
            if dependency not in by_id:
                raise ValueError(f"node {node.node_id!r} depends on missing node {dependency!r}.")
            children[dependency].add(node.node_id)
            indegree[node.node_id] += 1
    ready = deque(sorted(node_id for node_id, count in indegree.items() if count == 0))
    order: list[str] = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        raise ValueError("pipeline dependency graph contains a cycle.")
    return tuple(order)


def validate_pipeline(
    pipeline: PipelineDefinition,
    registry: OperationRegistry,
    *,
    dataset_store: DatasetStore | None = None,
    execution_records: dict[str, ExecutionRecord] | None = None,
    completed_node_ids: set[str] | None = None,
    available_artifact_references: set[str] | None = None,
    node_records: dict[str, ExecutionRecord] | None = None,
) -> PipelineValidation:
    """Validate graph structure, operation schemas, metadata and active sources."""
    records = execution_records or {}
    completed = set(completed_node_ids or ())
    for reused_node in pipeline.nodes:
        if not reused_node.reuse_execution_id:
            continue
        record = records.get(reused_node.reuse_execution_id)
        try:
            registered = registry.get(reused_node.operation_id, version=reused_node.operation_version)
        except (KeyError, ValueError):
            continue
        if (
            record is not None
            and record.status is ExecutionStatus.SUCCEEDED
            and record.operation_id == reused_node.operation_id
            and record.operation_version == reused_node.operation_version
            and record.implementation_version == registered.implementation_version
        ):
            completed.add(reused_node.node_id)
    errors: list[str] = []
    statuses: list[PipelineNodeValidation] = []
    try:
        order = _topological_order(pipeline.nodes)
    except ValueError as exc:
        order = ()
        errors.append(str(exc))
    by_id = {node.node_id: node for node in pipeline.nodes}
    for node in pipeline.nodes:
        reasons: list[str] = []
        warnings: list[str] = []
        if node.reuse_execution_id:
            record = records.get(node.reuse_execution_id)
            if record is None:
                reasons.append("selected historical execution is unavailable in this project.")
            elif record.status is not ExecutionStatus.SUCCEEDED:
                reasons.append("only successful historical executions can be reused.")
            elif record.operation_id != node.operation_id or record.operation_version != node.operation_version:
                reasons.append("historical execution operation identity/version does not match the node.")
            else:
                try:
                    reused_definition = registry.get(node.operation_id, version=node.operation_version)
                    if record.implementation_version != reused_definition.implementation_version:
                        reasons.append("historical execution implementation version is incompatible.")
                except (KeyError, ValueError) as exc:
                    reasons.append(str(exc))
            if record is not None and record.status is ExecutionStatus.SUCCEEDED:
                output = record.output
                artifact_reference = None
                if output is not None:
                    artifact_reference = getattr(output, "artifact_reference", None) or getattr(
                        output, "interval_artifact_reference", None
                    )
                if artifact_reference and (
                    available_artifact_references is None
                    or artifact_reference not in available_artifact_references
                ):
                    reasons.append("historical result artifact is unavailable or has unverified project lineage.")
            statuses.append(PipelineNodeValidation(
                node_id=node.node_id,
                status="reused" if not reasons else "blocked",
                reasons=tuple(reasons),
            ))
            if not reasons:
                completed.add(node.node_id)
            continue
        try:
            definition = registry.get(node.operation_id, version=node.operation_version)
        except (KeyError, ValueError) as exc:
            reasons.append(str(exc))
            statuses.append(PipelineNodeValidation(node_id=node.node_id, status="invalid", reasons=tuple(reasons)))
            continue
        warnings.extend(definition.warnings)
        warnings.extend(definition.limitations)
        try:
            definition.parameters_model.model_validate(node.parameters, strict=True)
        except (ValidationError, TypeError, ValueError) as exc:
            reasons.append(f"invalid parameters: {exc}")
        input_fields = definition.input_model.model_fields
        bindings_by_target: dict[str, list[PipelineInputBinding]] = defaultdict(list)
        resolved_inputs: dict[str, list[Any]] = defaultdict(list)
        unresolved_output_binding = False
        for binding in node.inputs:
            if binding.target_field not in input_fields:
                reasons.append(f"unknown input field {binding.target_field!r} for {node.operation_id}.")
                continue
            bindings_by_target[binding.target_field].append(binding)
            if binding.source_kind == "dataset":
                if binding.target_field != "signal":
                    reasons.append("dataset selections may bind only to a signal input.")
                    continue
                if dataset_store is None:
                    reasons.append("dataset binding requires an active compatible dataset registration.")
                    continue
                try:
                    registration = dataset_store.registration_for_dataset(binding.dataset)
                    if any(channel >= registration.channel_count for channel in binding.channels):
                        raise DatasetAccessError("selected channel is outside the active dataset.")
                    if binding.sample_interval.stop > registration.sample_count:
                        raise DatasetAccessError("sample interval exceeds the active dataset.")
                    reference = registration.reference(
                        binding.channels, binding.sample_interval, component_id=binding.component_id
                    )
                    resolved_inputs[binding.target_field].append(reference)
                    metadata = dataset_store.resolve(reference).metadata
                    for requirement in definition.metadata_requirements:
                        value = metadata.get(requirement.name)
                        if requirement.required and (
                            value is None or value.value is None or value.status.value == "unknown"
                        ):
                            reasons.append(f"required metadata {requirement.name!r} is missing or unknown.")
                        elif requirement.require_confirmed and value is not None and value.status.value != "confirmed":
                            reasons.append(f"metadata {requirement.name!r} is not confirmed.")
                except (DatasetAccessError, ValueError) as exc:
                    reasons.append(f"dataset input is unavailable or stale: {exc}")
            elif binding.source_kind == "node_output":
                source = by_id.get(binding.source_node_id or "")
                if source is None:
                    reasons.append(f"input references missing node {binding.source_node_id!r}.")
                    continue
                if source.node_id not in node.depends_on:
                    reasons.append(f"source node {source.node_id!r} is not declared as a dependency.")
                source_port = binding.source_port or ""
                edge = (source.operation_id, node.operation_id, binding.target_field, source_port)
                if edge not in _SUPPORTED_EDGE_TYPES:
                    reasons.append(f"unsupported operation connection: {source.operation_id} -> {node.operation_id}.")
                    unresolved_output_binding = True
                else:
                    source_record = (node_records or {}).get(source.node_id)
                    if source_record is None and source.reuse_execution_id:
                        source_record = records.get(source.reuse_execution_id)
                    if source_record is not None and source_record.status is ExecutionStatus.SUCCEEDED:
                        try:
                            if source_port == "detection_events":
                                collection = detection_collection_from_record(source_record)
                                adapted = (
                                    isi_collection_from_detection_collection(collection)
                                    if binding.target_field == "events"
                                    else collection
                                )
                            else:
                                adapted = mua_artifact_reference_from_record(source_record)
                            resolved_inputs[binding.target_field].append(adapted)
                        except (TypeError, ValueError, KeyError) as exc:
                            reasons.append(f"upstream output is incompatible: {exc}")
                            unresolved_output_binding = True
                    else:
                        reasons.append(
                            f"upstream result for node {source.node_id!r} is not available for compatibility validation."
                        )
                        unresolved_output_binding = True
            elif binding.source_kind == "artifact":
                if node.operation_id != MUA_AVERAGE_ID or binding.target_field != "traces":
                    reasons.append("only explicit MUA artifact inputs are supported as standalone artifact bindings.")
                else:
                    try:
                        import json

                        annotation = input_fields[binding.target_field].annotation
                        item_type = get_args(annotation)[0] if get_origin(annotation) is tuple else annotation
                        validated_artifact = TypeAdapter(item_type).validate_json(
                            json.dumps(binding.artifact_value, allow_nan=False), strict=True
                        )
                        resolved_inputs[binding.target_field].append(validated_artifact)
                    except (TypeError, ValueError, ValidationError) as exc:
                        reasons.append(f"artifact input is incompatible with {binding.target_field!r}: {exc}")
                    reference_value = binding.artifact_value
                    artifact_reference = (
                        reference_value.get("artifact_reference")
                        if isinstance(reference_value, dict)
                        else None
                    )
                    if isinstance(artifact_reference, str) and (
                        available_artifact_references is None
                        or artifact_reference not in available_artifact_references
                    ):
                        reasons.append("artifact reference is unavailable or has unverified project lineage.")
        for field_name, field in input_fields.items():
            if field.is_required() and field_name not in bindings_by_target:
                reasons.append(f"required operation input {field_name!r} has no binding.")
        actual_dependencies = {binding.source_node_id for binding in node.inputs if binding.source_kind == "node_output"}
        if actual_dependencies != set(node.depends_on):
            reasons.append("depends_on must exactly match the node-output input bindings.")
        dependency_operation_ids = {
            by_id[dependency].operation_id
            for dependency in node.depends_on
            if dependency in by_id
        }
        if not set(definition.prerequisites).issubset(dependency_operation_ids):
            reasons.append("required operation prerequisites are not supplied by upstream nodes.")
        pending_dependencies = set(node.depends_on) - completed
        if pending_dependencies:
            reasons.append(
                "upstream results are not yet available: "
                + ", ".join(sorted(pending_dependencies)) + "."
            )
        if node.operation_id in _DETECTORS | {CALC_MUA_ID}:
            selection = [item for item in node.inputs if item.target_field == "signal" and item.source_kind == "dataset"]
            if selection and len(selection[0].channels) != 1:
                reasons.append("this operation requires exactly one selected channel.")
        required_fields = {
            field_name for field_name, field in input_fields.items() if field.is_required()
        }
        artifact_bound_fields = {
            field_name for field_name, items in bindings_by_target.items()
            if items and all(item.source_kind == "artifact" for item in items)
        }
        if required_fields and required_fields.issubset(artifact_bound_fields):
            try:
                import json

                artifact_payload = {
                    field_name: [item.artifact_value for item in bindings_by_target[field_name]]
                    if field_name == "traces"
                    else bindings_by_target[field_name][0].artifact_value
                    for field_name in artifact_bound_fields
                }
                definition.input_model.model_validate_json(
                    json.dumps(artifact_payload, allow_nan=False), strict=True
                )
            except (TypeError, ValueError, ValidationError) as exc:
                reasons.append(f"artifact inputs fail the operation's compatibility contract: {exc}")
        if required_fields.issubset(resolved_inputs) and not unresolved_output_binding:
            candidate_input = {
                field_name: (
                    tuple(resolved_inputs[field_name])
                    if field_name == "traces"
                    else resolved_inputs[field_name][0]
                )
                for field_name in required_fields
            }
            try:
                definition.input_model.model_validate(candidate_input, strict=True)
            except (TypeError, ValueError, ValidationError) as exc:
                reasons.append(f"resolved inputs fail the operation schema or compatibility contract: {exc}")
        if reasons:
            availability_reasons = {
                reason for reason in reasons
                if "active compatible dataset" in reason
                or "unavailable or stale" in reason
                or "required metadata" in reason
                or "metadata '" in reason and "is not confirmed" in reason
                or "upstream results are not yet available" in reason
                or "upstream result for node" in reason
                or "artifact reference is unavailable" in reason
                or "historical result artifact is unavailable" in reason
            }
            status = "blocked" if len(availability_reasons) == len(reasons) else "invalid"
        else:
            status = "ready"
            if node.operation_id in _DETECTORS | {CALC_MUA_ID} and dataset_store is None:
                status = "blocked"
                reasons.append("node requires an active compatible dataset.")
        statuses.append(PipelineNodeValidation(node_id=node.node_id, status=status, reasons=tuple(reasons), warnings=tuple(warnings)))
    structurally_valid = not errors and all(item.status not in {"invalid"} for item in statuses)
    execution_ready = structurally_valid and any(item.status == "ready" for item in statuses)
    return PipelineValidation(
        pipeline_id=pipeline.pipeline_id,
        revision=pipeline.revision,
        structurally_valid=structurally_valid,
        execution_ready=execution_ready,
        topological_order=order,
        nodes=tuple(statuses),
        errors=tuple(errors),
    )


def _find_port_binding(node: PipelineNode, target: str, source_port: str | None = None) -> tuple[PipelineInputBinding, ...]:
    return tuple(
        item for item in node.inputs
        if item.target_field == target and (source_port is None or item.source_port == source_port)
    )


def _output_reference(record: ExecutionRecord) -> str:
    output = record.output
    if output is not None:
        for field in ("artifact_reference", "interval_artifact_reference"):
            value = getattr(output, field, None)
            if isinstance(value, str):
                return value
    return f"execution://{record.execution_id}"


class PipelineExecutionService:
    """Sequential pipeline coordinator; every scientific node still uses OperationRunner."""

    def __init__(self, registry: OperationRegistry, runner: OperationRunner, dataset_store: DatasetStore):
        self.registry = registry
        self.runner = runner
        self.dataset_store = dataset_store
        self._prepared: dict[tuple[str, str], PreparedExecution] = {}
        self._approved: dict[tuple[str, str], ApprovedExecution] = {}
        self._node_fingerprints: dict[tuple[str, str], str] = {}

    def start(self, pipeline: PipelineDefinition) -> PipelineRunState:
        now = datetime.now(timezone.utc)
        return PipelineRunState(
            run_id=str(uuid4()),
            pipeline_id=pipeline.pipeline_id,
            pipeline_revision=pipeline.revision,
            started_at_utc=now,
            updated_at_utc=now,
            node_status={node.node_id: "pending" for node in pipeline.nodes},
        )

    def bind_reused_nodes(
        self,
        pipeline: PipelineDefinition,
        state: PipelineRunState,
        historical_records: dict[str, ExecutionRecord],
        *,
        available_artifact_references: set[str] | None = None,
    ) -> PipelineRunState:
        """Bind selected historical results without creating approvals or executions."""
        if state.pipeline_id != pipeline.pipeline_id or state.pipeline_revision != pipeline.revision:
            raise ValueError("pipeline run state does not match this saved pipeline revision.")
        statuses = dict(state.node_status)
        execution_ids = dict(state.execution_ids)
        for node in pipeline.nodes:
            if not node.reuse_execution_id:
                continue
            record = historical_records.get(node.reuse_execution_id)
            definition = self.registry.get(node.operation_id, version=node.operation_version)
            if record is None or record.status is not ExecutionStatus.SUCCEEDED:
                raise ValueError(f"reused execution for node {node.node_id!r} is unavailable or unsuccessful.")
            if (record.operation_id, record.operation_version, record.implementation_version) != (
                node.operation_id, node.operation_version, definition.implementation_version
            ):
                raise ValueError(f"reused execution for node {node.node_id!r} has incompatible operation provenance.")
            output = record.output
            artifact_reference = None
            if output is not None:
                artifact_reference = getattr(output, "artifact_reference", None) or getattr(
                    output, "interval_artifact_reference", None
                )
            if artifact_reference and (
                available_artifact_references is None
                or artifact_reference not in available_artifact_references
            ):
                raise ValueError(f"reused execution for node {node.node_id!r} has no verified artifact available in this project.")
            statuses[node.node_id] = "reused"
            execution_ids[node.node_id] = record.execution_id
        payload = state.model_dump(mode="python")
        payload.update(updated_at_utc=datetime.now(timezone.utc), node_status=statuses, execution_ids=execution_ids)
        return PipelineRunState.model_validate(payload)

    def prepare_node(
        self,
        pipeline: PipelineDefinition,
        state: PipelineRunState,
        node_id: str,
        *,
        node_records: dict[str, ExecutionRecord],
        historical_records: dict[str, ExecutionRecord] | None = None,
    ) -> tuple[PreparedPipelineNode, PreparedExecution]:
        if state.pipeline_id != pipeline.pipeline_id or state.pipeline_revision != pipeline.revision:
            raise ValueError("pipeline run state does not match this saved pipeline revision.")
        node = next((item for item in pipeline.nodes if item.node_id == node_id), None)
        if node is None:
            raise KeyError(f"unknown pipeline node {node_id!r}.")
        if node.reuse_execution_id:
            raise ValueError("reused nodes do not prepare or execute a new operation.")
        failed_dependencies = [dep for dep in node.depends_on if state.node_status.get(dep) not in {"succeeded", "reused"}]
        if failed_dependencies:
            raise ValueError(f"pipeline node is blocked by incomplete dependencies: {', '.join(failed_dependencies)}.")
        if state.node_status.get(node_id) != "pending":
            raise ValueError("pipeline node is not pending.")
        definition = self.registry.get(node.operation_id, version=node.operation_version)
        bindings: dict[str, list[Any]] = defaultdict(list)
        metadata: tuple[MetadataSnapshot, ...] = ()
        dataset_ref: DatasetReference | None = None
        selected_channels: tuple[int, ...] = ()
        sample_interval: SampleInterval | None = None
        for binding in node.inputs:
            if binding.source_kind == "dataset":
                registration = self.dataset_store.registration_for_dataset(binding.dataset)
                data_ref = registration.reference(binding.channels, binding.sample_interval, component_id=binding.component_id)
                resolved = self.dataset_store.resolve(data_ref)
                dataset_ref = registration.dataset
                selected_channels = binding.channels
                sample_interval = binding.sample_interval
                metadata = tuple(
                    MetadataSnapshot(name=name, value=value.value, status=value.status)
                    for name, value in resolved.metadata.items()
                )
                bindings[binding.target_field].append(data_ref)
            elif binding.source_kind == "artifact":
                import json

                annotation = definition.input_model.model_fields[binding.target_field].annotation
                item_type = get_args(annotation)[0] if get_origin(annotation) is tuple else annotation
                typed_artifact = TypeAdapter(item_type).validate_json(
                    json.dumps(binding.artifact_value, allow_nan=False), strict=True
                )
                bindings[binding.target_field].append(typed_artifact)
            else:
                source_record = node_records.get(binding.source_node_id or "")
                if source_record is None:
                    source_node = next(item for item in pipeline.nodes if item.node_id == binding.source_node_id)
                    source_record = (historical_records or {}).get(source_node.reuse_execution_id or "")
                if source_record is None or source_record.status is not ExecutionStatus.SUCCEEDED:
                    raise ValueError(f"upstream result for node {binding.source_node_id!r} is unavailable.")
                if binding.source_port == "detection_events":
                    collection = detection_collection_from_record(source_record)
                    if binding.target_field == "events":
                        adapted = isi_collection_from_detection_collection(collection)
                    elif binding.target_field == "detections":
                        adapted = collection
                    else:
                        raise ValueError("detection event output is incompatible with this input field.")
                    bindings[binding.target_field].append(adapted)
                elif binding.source_port == "mua_trace":
                    if binding.target_field != "traces":
                        raise ValueError("MUA artifact output is compatible only with the traces input.")
                    adapted = mua_artifact_reference_from_record(source_record)
                    bindings[binding.target_field].append(adapted)
        input_payload: dict[str, Any] = {}
        for field, values in bindings.items():
            input_payload[field] = tuple(values) if field == "traces" else values[0]
        normalized_input = definition.input_model.model_validate(input_payload, strict=True)
        prerequisites = []
        for dependency in node.depends_on:
            source_record = node_records.get(dependency)
            source_node = next(item for item in pipeline.nodes if item.node_id == dependency)
            if source_record is None:
                source_record = (historical_records or {}).get(source_node.reuse_execution_id or "")
            if source_record and source_record.operation_id in definition.prerequisites:
                prerequisites.append(PrerequisiteReference(
                    execution_id=source_record.execution_id,
                    operation_id=source_record.operation_id,
                    operation_version=source_record.operation_version,
                    output_reference=_output_reference(source_record),
                    status=source_record.status,
                ))
        context = OperationContext(
            dataset_reference=dataset_ref,
            metadata=metadata,
            selected_channels=selected_channels,
            sample_interval=sample_interval,
            prerequisites=tuple(prerequisites),
        )
        prepared = self.runner.prepare(
            node.operation_id, node.operation_version, normalized_input, node.parameters, context=context
        )
        token = (state.run_id, node_id)
        if token in self._prepared:
            raise ValueError("this pipeline node already has a pending preparation.")
        self._prepared[token] = prepared
        self._node_fingerprints[token] = hashlib.sha256(node.model_dump_json().encode("utf-8")).hexdigest()
        return (
            PreparedPipelineNode(
                run_id=state.run_id,
                node_id=node_id,
                operation_id=prepared.operation_id,
                operation_version=prepared.operation_version,
                implementation_version=prepared.implementation_version,
                request_snapshot=prepared.to_snapshot(),
                warnings=prepared.warnings,
            ),
            prepared,
        )

    def approve_node(self, run_id: str, node_id: str, *, approved_by: str) -> ApprovedExecution:
        token = (run_id, node_id)
        prepared = self._prepared.get(token)
        if prepared is None:
            raise ApprovalRequiredError("pipeline node must be prepared before approval.")
        approved = self.runner.approve(prepared, approved_by=approved_by)
        self._approved[token] = approved
        return approved

    def discard_prepared(self, run_id: str, node_id: str) -> None:
        """Dismiss a reviewed node proposal; it cannot be executed afterward."""
        token = (run_id, node_id)
        self._prepared.pop(token, None)
        self._approved.pop(token, None)
        self._node_fingerprints.pop(token, None)

    def execute_node(
        self,
        pipeline: PipelineDefinition,
        state: PipelineRunState,
        node_id: str,
        approved: ApprovedExecution,
        *,
        project_store: Any | None = None,
    ) -> tuple[PipelineRunState, ExecutionRecord]:
        token = (state.run_id, node_id)
        if self._approved.get(token) is not approved:
            raise ApprovalRequiredError("pipeline node requires its explicit runner-issued approval.")
        if node_id not in state.node_status or state.node_status[node_id] not in {"pending", "prepared"}:
            raise ValueError("pipeline node is not pending or prepared.")
        if state.pipeline_id != pipeline.pipeline_id or state.pipeline_revision != pipeline.revision:
            raise ValueError("pipeline run state does not match this saved pipeline revision.")
        current_node = next((item for item in pipeline.nodes if item.node_id == node_id), None)
        expected_fingerprint = self._node_fingerprints.get(token)
        if current_node is None or expected_fingerprint != hashlib.sha256(current_node.model_dump_json().encode("utf-8")).hexdigest():
            self._prepared.pop(token, None)
            self._approved.pop(token, None)
            self._node_fingerprints.pop(token, None)
            raise ValueError("pipeline node changed after preparation; prepare it again for review.")
        record = self.runner.execute(approved)
        if project_store is not None:
            project_store.append_execution_record(record)
        statuses = dict(state.node_status)
        ids = dict(state.execution_ids)
        errors = dict(state.errors)
        statuses[node_id] = "succeeded" if record.status is ExecutionStatus.SUCCEEDED else "failed"
        ids[node_id] = record.execution_id
        if record.error is not None:
            errors[node_id] = f"{record.error.error_type}: {record.error.message}"
        if record.status is ExecutionStatus.FAILED:
            blocked = {node_id}
            changed = True
            while changed:
                changed = False
                for dependent in pipeline.nodes:
                    if dependent.node_id not in blocked and blocked.intersection(dependent.depends_on):
                        blocked.add(dependent.node_id)
                        changed = True
            for blocked_id in blocked - {node_id}:
                if statuses.get(blocked_id) == "pending":
                    statuses[blocked_id] = "blocked"
        state_payload = state.model_dump(mode="python")
        state_payload.update(
            updated_at_utc=datetime.now(timezone.utc),
            node_status=statuses,
            execution_ids=ids,
            errors=errors,
        )
        next_state = PipelineRunState.model_validate(state_payload)
        if project_store is not None:
            project_store.save_pipeline_run(next_state)
        self._prepared.pop(token, None)
        self._approved.pop(token, None)
        self._node_fingerprints.pop(token, None)
        return next_state, record

    @staticmethod
    def deterministic_order(pipeline: PipelineDefinition) -> tuple[str, ...]:
        return _topological_order(pipeline.nodes)

