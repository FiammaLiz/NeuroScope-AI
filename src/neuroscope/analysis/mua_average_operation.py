"""Approved equal-channel averaging of compatible project MUAtrace artifacts.

The average composes stored calcMUA outputs and does not change their upstream
scientific method or make an independent population-activity estimate.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from math import isfinite
from typing import Literal

import numpy as np
from pydantic import Field, StrictInt, field_validator, model_validator

from ..exploration import MetadataStatus
from ..mua_artifacts import MUAArtifactError, MUAArtifactStore
from .contracts import (
    ExecutionStatus,
    OperationContext,
    OperationDefinition,
    OperationModel,
    OperationResult,
    OperationRuntimeAccess,
    SampleInterval,
)
from .mua_operation import OPERATION_ID as CALC_MUA_ID
from .mua_operation import CalcMUAOutput
from .operations import ExecutionRecord
from .registry import OperationRegistry

OPERATION_ID = "neuroscope.mua.average-traces"
OPERATION_VERSION = "1.0.0"
IMPLEMENTATION_VERSION = "equal-channel-mean-v1"
AVERAGING_METHOD = "pointwise arithmetic mean; equal channel weights"


class MUAArtifactReference(OperationModel):
    """Serializable scientific identity and compatibility metadata for one trace."""

    artifact_reference: str = Field(pattern=r"^artifact://mua/[0-9a-f-]{36}$")
    execution_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    channel_index: StrictInt = Field(ge=0)
    sample_interval: SampleInterval
    output_length: StrictInt = Field(gt=0)
    sampling_rate_hz: float = Field(gt=0, allow_inf_nan=False)
    sampling_rate_status: MetadataStatus
    mua_units: Literal["s^-1", "events s^-1"]
    amplitude_units: str | None
    amplitude_units_status: MetadataStatus
    mode: Literal["rel", "abs"]
    stdmin: float = Field(allow_inf_nan=False)
    actual_threshold: float = Field(allow_inf_nan=False)
    noise_estimate: float = Field(ge=0, allow_inf_nan=False)
    bandwidth_s: float = Field(gt=0, allow_inf_nan=False)
    num_points: StrictInt = Field(gt=0)
    source_sha256: str = Field(min_length=1)
    source_operation_id: str = Field(min_length=1)
    source_operation_version: str = Field(min_length=1)
    source_implementation_version: str = Field(min_length=1)
    signal_kind: str | None
    signal_kind_status: MetadataStatus
    filtering_description: str | None
    filtering_status: MetadataStatus

    @field_validator(
        "sampling_rate_hz", "stdmin", "actual_threshold", "noise_estimate", "bandwidth_s"
    )
    @classmethod
    def finite_values(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("MUA artifact metadata must be finite.")
        return value


class MUAAverageInput(OperationModel):
    """Explicit artifact inputs; no GUI selection or numerical arrays are embedded."""

    traces: tuple[MUAArtifactReference, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def distinct_inputs(self) -> MUAAverageInput:
        references = [item.artifact_reference for item in self.traces]
        executions = [item.execution_id for item in self.traces]
        channels = [item.channel_index for item in self.traces]
        if len(set(references)) != len(references):
            raise ValueError("MUA averaging inputs must not repeat an artifact.")
        if len(set(executions)) != len(executions):
            raise ValueError("MUA averaging inputs must not repeat an execution.")
        if len(set(channels)) != len(channels):
            raise ValueError("MUA averaging inputs must come from distinct channels.")
        _validate_compatible(self.traces)
        return self


class MUAAverageParameters(OperationModel):
    """Versioned fixed method; averaging is not configurable in version 1."""

    method: Literal["equal_channel_arithmetic_mean"] = "equal_channel_arithmetic_mean"
    block_samples: StrictInt = Field(default=65536, gt=0)


class MUAAverageOutput(OperationModel):
    artifact_reference: str = Field(pattern=r"^artifact://mua/[0-9a-f-]{36}$")
    dataset_id: str = Field(min_length=1)
    channel_indices: tuple[StrictInt, ...] = Field(min_length=2)
    source_artifact_references: tuple[str, ...] = Field(min_length=2)
    source_execution_ids: tuple[str, ...] = Field(min_length=2)
    source_thresholds: tuple[float, ...] = Field(min_length=2)
    source_noise_estimates: tuple[float, ...] = Field(min_length=2)
    contributing_channels: StrictInt = Field(ge=2)
    sample_interval: SampleInterval
    output_length: StrictInt = Field(gt=0)
    sampling_rate_hz: float = Field(gt=0, allow_inf_nan=False)
    sampling_rate_status: MetadataStatus
    mua_units: Literal["s^-1", "events s^-1"]
    method: Literal["pointwise arithmetic mean; equal channel weights"] = AVERAGING_METHOD
    method_version: str = OPERATION_VERSION
    mode: Literal["rel", "abs"]
    stdmin: float = Field(allow_inf_nan=False)
    bandwidth_s: float = Field(gt=0, allow_inf_nan=False)
    num_points: StrictInt = Field(gt=0)
    source_sha256: str = Field(min_length=1)
    signal_kind: str | None
    signal_kind_status: MetadataStatus
    amplitude_units: str | None
    amplitude_units_status: MetadataStatus
    filtering_description: str | None
    filtering_status: MetadataStatus
    execution_id: str = Field(min_length=1)
    approval_id: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    executed_at_utc: datetime

    @model_validator(mode="after")
    def aligned_sources(self) -> MUAAverageOutput:
        if not (
            len(self.channel_indices)
            == len(self.source_artifact_references)
            == len(self.source_execution_ids)
            == len(self.source_thresholds)
            == len(self.source_noise_estimates)
            == self.contributing_channels
        ):
            raise ValueError("averaged output source lists must match contributing_channels.")
        if any(not isfinite(value) for value in self.source_thresholds):
            raise ValueError("source threshold snapshots must be finite.")
        if any(not isfinite(value) or value < 0 for value in self.source_noise_estimates):
            raise ValueError("source noise snapshots must be finite and non-negative.")
        return self


def _validate_compatible(traces: Sequence[MUAArtifactReference]) -> None:
    """Reject non-identical scientific grids, method parameters, and source context."""
    first = traces[0]
    if first.sampling_rate_status is not MetadataStatus.CONFIRMED:
        raise ValueError("MUA averaging requires confirmed sampling rates for every input.")
    if first.amplitude_units_status is not MetadataStatus.CONFIRMED or not first.amplitude_units:
        raise ValueError("MUA averaging requires confirmed source amplitude units for every input.")
    if first.signal_kind_status is not MetadataStatus.CONFIRMED or not first.signal_kind:
        raise ValueError("MUA averaging requires confirmed source signal kind for every input.")
    if first.filtering_status is not MetadataStatus.CONFIRMED or not first.filtering_description:
        raise ValueError(
            "MUA averaging requires confirmed filtering/preprocessing information for every input."
        )
    for index, trace in enumerate(traces[1:], start=2):
        fields = (
            "dataset_id", "sample_interval", "output_length", "sampling_rate_hz",
            "sampling_rate_status", "mua_units", "amplitude_units",
            "amplitude_units_status", "mode", "stdmin", "bandwidth_s", "num_points",
            "source_sha256", "source_operation_id", "source_operation_version",
            "source_implementation_version", "signal_kind", "signal_kind_status",
            "filtering_description", "filtering_status",
        )
        for field in fields:
            if getattr(first, field) != getattr(trace, field):
                raise ValueError(f"MUA trace input {index} is incompatible: {field} differs.")
        if trace.sampling_rate_status is not MetadataStatus.CONFIRMED:
            raise ValueError(f"MUA trace input {index} has no confirmed sampling rate.")


def _validate_artifact_manifest(
    trace: MUAArtifactReference, manifest: dict[str, object]
) -> None:
    expected = {
        "artifact_id": trace.artifact_reference.rsplit("/", 1)[-1],
        "execution_id": trace.execution_id,
        "dataset_id": trace.dataset_id,
        "source_channel": trace.channel_index,
        "sample_interval": trace.sample_interval.model_dump(mode="json"),
        "sampling_rate_hz": trace.sampling_rate_hz,
        "sampling_rate_status": trace.sampling_rate_status.value,
        "mua_units": trace.mua_units,
        "input_amplitude_units": trace.amplitude_units,
        "input_amplitude_units_status": trace.amplitude_units_status.value,
        "parameters": {
            "stdmin": trace.stdmin,
            "mua_window_s": trace.bandwidth_s,
            "mode": trace.mode,
            "num_points": trace.num_points,
        },
        "source_sha256": trace.source_sha256,
        "input_signal_kind": trace.signal_kind,
        "input_signal_kind_status": trace.signal_kind_status.value,
        "filtering_description": trace.filtering_description,
        "filtering_status": trace.filtering_status.value,
        "implementation_version": trace.source_implementation_version,
        "operation_id": trace.source_operation_id,
        "operation_version": trace.source_operation_version,
        "threshold": trace.actual_threshold,
        "noise_estimate": trace.noise_estimate,
    }
    for field, value in expected.items():
        if field in {"operation_id", "operation_version"} and field not in manifest:
            # Task 13 artifacts predate semantic operation ID/version in their manifests;
            # the persisted typed execution record remains the source of that identity.
            continue
        if manifest.get(field) != value:
            raise MUAArtifactError(f"artifact metadata no longer matches its approved reference ({field}).")
    if manifest.get("method") != "MATLAB calcmua_verFi.m":
        raise MUAArtifactError("input artifact is not an original calcMUA trace.")


def mua_artifact_reference_from_record(record: ExecutionRecord) -> MUAArtifactReference:
    """Create a downstream-safe artifact reference from a successful calcMUA record."""
    if record.status is not ExecutionStatus.SUCCEEDED:
        raise ValueError("only successful executions can supply an MUA artifact reference.")
    if record.operation_id != CALC_MUA_ID or not isinstance(record.output, CalcMUAOutput):
        raise TypeError("execution record does not contain a supported calcMUA output.")
    output = record.output
    metadata = {item.name: item for item in record.input_metadata}
    signal_kind = metadata.get("signal_kind")
    return MUAArtifactReference(
        artifact_reference=output.artifact_reference,
        execution_id=record.execution_id,
        dataset_id=output.dataset_id,
        channel_index=output.source_channel,
        sample_interval=output.sample_interval,
        output_length=output.output_length,
        sampling_rate_hz=output.sampling_rate_hz,
        sampling_rate_status=output.sampling_rate_status,
        mua_units=output.mua_units,
        amplitude_units=output.amplitude_units,
        amplitude_units_status=output.amplitude_units_status,
        mode=output.mode,
        stdmin=output.stdmin,
        bandwidth_s=output.bandwidth_s,
        num_points=output.num_points,
        source_sha256=output.source_sha256,
        source_operation_id=record.operation_id,
        source_operation_version=record.operation_version,
        source_implementation_version=record.implementation_version,
        actual_threshold=output.threshold,
        noise_estimate=output.noise_estimate,
        signal_kind=(str(signal_kind.value) if signal_kind and signal_kind.value is not None else None),
        signal_kind_status=(signal_kind.status if signal_kind else MetadataStatus.UNKNOWN),
        filtering_description=output.filtering_description,
        filtering_status=output.filtering_status,
    )


def mua_artifact_reference_from_snapshot(snapshot: dict[str, object]) -> MUAArtifactReference:
    """Rebuild a typed calcMUA artifact handle from a persisted project record."""
    if snapshot.get("status") != ExecutionStatus.SUCCEEDED.value:
        raise ValueError("only successful persisted executions can supply an MUA artifact.")
    if snapshot.get("operation_id") != CALC_MUA_ID:
        raise TypeError("persisted execution is not a calcMUA operation.")
    output = CalcMUAOutput.model_validate_json(
        json.dumps(snapshot.get("output"), allow_nan=False), strict=True
    )
    context = snapshot.get("context")
    if not isinstance(context, dict) or not isinstance(context.get("metadata"), list):
        raise TypeError("persisted calcMUA execution has no metadata snapshot.")
    metadata = {
        item["name"]: item
        for item in context["metadata"]
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    signal_kind = metadata.get("signal_kind", {})
    try:
        return MUAArtifactReference(
            artifact_reference=output.artifact_reference,
            execution_id=str(snapshot["execution_id"]),
            dataset_id=output.dataset_id,
            channel_index=output.source_channel,
            sample_interval=output.sample_interval,
            output_length=output.output_length,
            sampling_rate_hz=output.sampling_rate_hz,
            sampling_rate_status=output.sampling_rate_status,
            mua_units=output.mua_units,
            amplitude_units=output.amplitude_units,
            amplitude_units_status=output.amplitude_units_status,
            mode=output.mode,
            stdmin=output.stdmin,
            actual_threshold=output.threshold,
            noise_estimate=output.noise_estimate,
            bandwidth_s=output.bandwidth_s,
            num_points=output.num_points,
            source_sha256=output.source_sha256,
            source_operation_id=CALC_MUA_ID,
            source_operation_version=str(snapshot["operation_version"]),
            source_implementation_version=str(snapshot["implementation_version"]),
            signal_kind=signal_kind.get("value"),
            signal_kind_status=MetadataStatus(signal_kind.get("status", "unknown")),
            filtering_description=output.filtering_description,
            filtering_status=output.filtering_status,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"persisted calcMUA execution has invalid artifact metadata: {exc}") from exc


def _execute_average(
    operation_input: MUAAverageInput,
    parameters: MUAAverageParameters,
    context: OperationContext,
    runtime: OperationRuntimeAccess,
) -> OperationResult[MUAAverageOutput]:
    del context
    if runtime.mua_artifact_store is None or not isinstance(
        runtime.mua_artifact_store, MUAArtifactStore
    ):
        raise RuntimeError("MUA averaging requires a project-owned MUAArtifactStore.")
    # Canonical order makes floating-point accumulation independent of selection order.
    traces = tuple(sorted(operation_input.traces, key=lambda item: item.artifact_reference))
    _validate_compatible(traces)
    opened = []
    try:
        for trace in traces:
            artifact = runtime.mua_artifact_store.open(trace.artifact_reference)
            opened.append(artifact)
            _validate_artifact_manifest(trace, artifact.manifest)
            if artifact.mua.shape != (trace.output_length,) or artifact.times_so.shape != artifact.mua.shape:
                raise MUAArtifactError("MUA artifact trace and time grid disagree with the approved input.")
        times = opened[0].times_so
        reference = traces[0]
        for start in range(0, reference.output_length, parameters.block_samples):
            stop = min(start + parameters.block_samples, reference.output_length)
            for artifact in opened[1:]:
                if not np.array_equal(times[start:stop], artifact.times_so[start:stop]):
                    raise ValueError("MUA averaging inputs have mismatched temporal coordinate arrays.")
        artifact_reference = runtime.mua_artifact_store.write_average(
            traces=tuple(artifact.mua for artifact in opened),
            times_so=times,
            manifest={
                "artifact_type": "averaged_mua_trace",
                "method": AVERAGING_METHOD,
                "method_version": OPERATION_VERSION,
                "operation_id": OPERATION_ID,
                "operation_version": OPERATION_VERSION,
                "implementation_version": IMPLEMENTATION_VERSION,
                "execution_id": runtime.execution_id,
                "approval_id": runtime.approval_id,
                "approved_by": runtime.approved_by,
                "executed_at_utc": runtime.started_at_utc.isoformat(),
                "dataset_id": reference.dataset_id,
                "source_channels": [item.channel_index for item in traces],
                "source_artifact_references": [item.artifact_reference for item in traces],
                "source_execution_ids": [item.execution_id for item in traces],
                "source_thresholds": [item.actual_threshold for item in traces],
                "source_noise_estimates": [item.noise_estimate for item in traces],
                "source_result_snapshots": [
                    {
                        "artifact_reference": item.artifact_reference,
                        "execution_id": item.execution_id,
                        "channel_index": item.channel_index,
                        "threshold": item.actual_threshold,
                        "noise_estimate": item.noise_estimate,
                        "parameters": {
                            "stdmin": item.stdmin,
                            "mua_window_s": item.bandwidth_s,
                            "mode": item.mode,
                            "num_points": item.num_points,
                        },
                    }
                    for item in traces
                ],
                "source_operation_id": reference.source_operation_id,
                "source_operation_version": reference.source_operation_version,
                "source_implementation_version": reference.source_implementation_version,
                "source_sha256": reference.source_sha256,
                "sample_interval": reference.sample_interval.model_dump(mode="json"),
                "sample_coordinate": "zero-based original recording sample; half-open selection",
                "sampling_rate_hz": reference.sampling_rate_hz,
                "sampling_rate_status": reference.sampling_rate_status.value,
                "mua_units": reference.mua_units,
                "mode": reference.mode,
                "stdmin": reference.stdmin,
                "mua_window_s": reference.bandwidth_s,
                "num_points": reference.num_points,
                "signal_kind": reference.signal_kind,
                "signal_kind_status": reference.signal_kind_status.value,
                "amplitude_units": reference.amplitude_units,
                "amplitude_units_status": reference.amplitude_units_status.value,
                "filtering_description": reference.filtering_description,
                "filtering_status": reference.filtering_status.value,
                "contributing_channels": len(traces),
                "weighting": "equal",
                "parameters": parameters.model_dump(mode="json"),
            },
            block_samples=parameters.block_samples,
        )
    finally:
        for artifact in opened:
            artifact.close()
    output = MUAAverageOutput(
        artifact_reference=artifact_reference,
        dataset_id=reference.dataset_id,
        channel_indices=tuple(item.channel_index for item in traces),
        source_artifact_references=tuple(item.artifact_reference for item in traces),
        source_execution_ids=tuple(item.execution_id for item in traces),
        source_thresholds=tuple(item.actual_threshold for item in traces),
        source_noise_estimates=tuple(item.noise_estimate for item in traces),
        contributing_channels=len(traces),
        sample_interval=reference.sample_interval,
        output_length=reference.output_length,
        sampling_rate_hz=reference.sampling_rate_hz,
        sampling_rate_status=reference.sampling_rate_status,
        mua_units=reference.mua_units,
        mode=reference.mode,
        stdmin=reference.stdmin,
        bandwidth_s=reference.bandwidth_s,
        num_points=reference.num_points,
        source_sha256=reference.source_sha256,
        signal_kind=reference.signal_kind,
        signal_kind_status=reference.signal_kind_status,
        amplitude_units=reference.amplitude_units,
        amplitude_units_status=reference.amplitude_units_status,
        filtering_description=reference.filtering_description,
        filtering_status=reference.filtering_status,
        execution_id=runtime.execution_id,
        approval_id=runtime.approval_id,
        approved_by=runtime.approved_by,
        executed_at_utc=runtime.started_at_utc,
    )
    return OperationResult(
        output=output,
        warnings=(
            "This is an equal-channel arithmetic mean of calcMUA artifacts; it is not an independent estimate of neuronal population activity.",
            "The average is defined only for finite input traces; NaN traces are rejected rather than dropped.",
        ),
    )


AVERAGE_MUA_OPERATION = OperationDefinition(
    operation_id=OPERATION_ID,
    name="Average compatible calcMUA traces",
    description="Create a new persistent, equal-channel mean from explicitly selected calcMUA artifacts.",
    version=OPERATION_VERSION,
    implementation_version=IMPLEMENTATION_VERSION,
    input_model=MUAAverageInput,
    parameters_model=MUAAverageParameters,
    output_model=MUAAverageOutput,
    execute=_execute_average,
    warnings=(
        "Averaging uses equal channel weights and does not establish an independent neuronal population estimate.",
    ),
    limitations=(
        "Input traces must have identical dataset, sample interval, time grid, confirmed sampling rate, calcMUA parameters, units, signal metadata, and filtering metadata.",
        "Nonfinite input traces are rejected; values are never interpolated, replaced, or omitted.",
    ),
)


def register_mua_average(registry: OperationRegistry) -> OperationDefinition:
    if not isinstance(registry, OperationRegistry):
        raise TypeError("registry must be an OperationRegistry.")
    registry.register(AVERAGE_MUA_OPERATION)
    return AVERAGE_MUA_OPERATION
