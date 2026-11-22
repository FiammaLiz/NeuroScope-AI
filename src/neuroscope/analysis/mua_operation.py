"""Approved operation adapter for NeuroScope's calcMUA implementation.

The scientific method was developed by Dr. Cecilia Herbert. Fiamma Liz Leites
independently implemented it in MATLAB as ``calcmua_verFi.m`` using the method
as a reference. NeuroScope's Python implementation is separate and based on
Leites' MATLAB implementation; MATLAB numerical parity remains unverified.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from typing import Literal

import numpy as np
from pydantic import Field, StrictInt, field_validator, model_validator

from ..exploration import MetadataStatus, MetadataValue
from ..mua_artifacts import MUAArtifactStore
from .contracts import (
    DataReference,
    MetadataRequirement,
    MetadataSnapshot,
    OperationContext,
    OperationDefinition,
    OperationModel,
    OperationResult,
    OperationRuntimeAccess,
    SampleInterval,
)
from .mua import (
    AMPLITUDE_CAP_FACTOR,
    NOISE_DIVISOR,
    PROMINENCE_FACTOR,
    calc_mua,
)
from .mua import IMPLEMENTATION_VERSION as CALC_IMPLEMENTATION_VERSION
from .registry import OperationRegistry

OPERATION_ID = "neuroscope.mua.calc-mua"
OPERATION_VERSION = "1.0.0"
IMPLEMENTATION_VERSION = CALC_IMPLEMENTATION_VERSION
# Retained historical provenance value. The original MATLAB file is not present
# here for independent digest re-verification; do not treat this comment as a
# newly verified source check.
SOURCE_SHA256 = "73047a291dfb8093e2404d6ae07e31b6d0c8b5e9e3ca315805fce6bd66e31cba"
_METADATA_FIELDS = (
    "signal_kind",
    "sampling_rate_hz",
    "units",
    "filtering_description",
)


class CalcMUAInput(OperationModel):
    """One selected continuous-voltage channel and half-open sample interval."""

    signal: DataReference

    @model_validator(mode="after")
    def one_channel(self) -> CalcMUAInput:
        if len(self.signal.channels) != 1:
            raise ValueError("calcMUA requires exactly one selected channel.")
        return self


class CalcMUAParameters(OperationModel):
    """Explicit source parameters; none are assigned implicit scientific defaults."""

    stdmin: float = Field(allow_inf_nan=False)
    mua_window_s: float = Field(gt=0, allow_inf_nan=False)
    mode: Literal["rel", "abs"]
    num_points: StrictInt = Field(gt=0)

    @field_validator("stdmin", "mua_window_s", mode="before")
    @classmethod
    def finite_real(cls, value: object, info) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{info.field_name} must be a finite real number.")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{info.field_name} must be finite.")
        return result


class CalcMUAOutput(OperationModel):
    """Compact scientific output; arrays and event lists live in an artifact."""

    artifact_reference: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    source_channel: int = Field(ge=0, strict=True)
    sample_interval: SampleInterval
    output_length: StrictInt = Field(gt=0)
    event_count: StrictInt = Field(ge=0)
    threshold: float = Field(allow_inf_nan=False)
    noise_estimate: float = Field(ge=0, allow_inf_nan=False)
    stdmin: float = Field(allow_inf_nan=False)
    bandwidth_s: float = Field(gt=0, allow_inf_nan=False)
    num_points: StrictInt = Field(gt=0)
    mode: Literal["rel", "abs"]
    mua_units: Literal["s^-1", "events s^-1"]
    amplitude_units: str | None = None
    amplitude_units_status: MetadataStatus
    sampling_rate_hz: float = Field(gt=0, allow_inf_nan=False)
    sampling_rate_status: MetadataStatus
    filtering_description: str | None = None
    filtering_status: MetadataStatus
    time_reference: Literal["selected_window_matlab_one_based"] = (
        "selected_window_matlab_one_based"
    )
    source_sha256: str
    algorithm_constants: dict[str, float]
    dependency_versions: dict[str, str]
    execution_id: str = Field(min_length=1)


def _metadata_snapshot(
    name: str,
    resolved: dict[str, MetadataValue],
    requested: dict[str, MetadataSnapshot],
) -> MetadataValue:
    actual = resolved[name]
    declared = requested.get(name)
    if declared is None:
        if actual.status is MetadataStatus.UNKNOWN and actual.value is None:
            return actual
        raise ValueError(f"execution context must snapshot metadata {name!r} and its status.")
    if actual.status is not declared.status or actual.value != declared.value:
        raise ValueError(f"execution context metadata {name!r} does not match the dataset.")
    return actual


def _prepare_warnings(
    operation_input: CalcMUAInput,
    _parameters: CalcMUAParameters,
    context: OperationContext,
) -> tuple[str, ...]:
    reference = operation_input.signal
    if context.dataset_reference != reference.dataset:
        raise ValueError("operation context dataset identity must match the calcMUA input reference.")
    if context.selected_channels != reference.channels:
        raise ValueError("operation context channels must match the calcMUA input reference.")
    if context.sample_interval != reference.sample_interval:
        raise ValueError("operation context sample interval must match the calcMUA input reference.")
    metadata = {item.name: item for item in context.metadata}
    filtering = metadata.get("filtering_description")
    if filtering is None or filtering.status is not MetadataStatus.CONFIRMED or not filtering.value:
        return (
            (
                "The source names its input filtered_spike_data, but filtering is not confirmed; "
                "calcMUA does not filter or modify the signal."
            ),
        )
    return ()


def _execute_calc_mua(
    input_data: CalcMUAInput,
    parameters: CalcMUAParameters,
    context: OperationContext,
    runtime: OperationRuntimeAccess,
) -> OperationResult[CalcMUAOutput]:
    reference = input_data.signal
    resolved = runtime.data_for(reference)
    if len(resolved.channels) != 1:
        raise ValueError("runtime access must contain exactly the referenced channel.")
    channel_index = reference.channels[0]
    signal = resolved.channel(channel_index).samples
    if signal.ndim != 1:
        raise ValueError("calcMUA requires a one-dimensional selected channel.")

    actual_metadata = dict(resolved.metadata)
    requested_metadata = {item.name: item for item in context.metadata}
    metadata = {
        name: _metadata_snapshot(name, actual_metadata, requested_metadata)
        for name in _METADATA_FIELDS
    }
    signal_kind = metadata["signal_kind"]
    if signal_kind.status is not MetadataStatus.CONFIRMED or signal_kind.value != "raw_voltage":
        raise ValueError("calcMUA requires confirmed signal_kind='raw_voltage'.")
    sampling_rate = metadata["sampling_rate_hz"]
    if (
        sampling_rate.status is not MetadataStatus.CONFIRMED
        or sampling_rate.value is None
    ):
        raise ValueError("calcMUA requires a confirmed sampling rate in Hz.")
    rate = float(sampling_rate.value)

    result = calc_mua(
        signal,
        sampling_rate_hz=rate,
        stdmin=parameters.stdmin,
        mua_window_s=parameters.mua_window_s,
        mode=parameters.mode,
        num_points=parameters.num_points,
    )
    if runtime.mua_artifact_store is None:
        raise RuntimeError("calcMUA requires a project-owned MUAArtifactStore.")
    if not isinstance(runtime.mua_artifact_store, MUAArtifactStore):
        raise TypeError("runtime MUA artifact service is not an MUAArtifactStore.")

    relative_indices = result.spike_sample_indices[0]
    source_indices = relative_indices + reference.sample_interval.start
    filter_metadata = metadata["filtering_description"]
    units_metadata = metadata["units"]
    amplitude_units = (
        str(units_metadata.value) if units_metadata.value is not None else None
    )
    manifest = {
        "operation_id": OPERATION_ID,
        "operation_version": OPERATION_VERSION,
        "method": "MATLAB calcmua_verFi.m",
        "source_sha256": SOURCE_SHA256,
        "implementation_version": IMPLEMENTATION_VERSION,
        "execution_id": runtime.execution_id,
        "dataset_id": reference.dataset.dataset_id,
        "source_channel": channel_index,
        "sample_interval": reference.sample_interval.model_dump(mode="json"),
        "sample_coordinate": "zero-based original recording sample; half-open selection",
        "sampling_rate_hz": rate,
        "sampling_rate_status": sampling_rate.status.value,
        "time_reference": "selected-window MATLAB grid (1..N)/sampling_rate_hz",
        "input_signal_kind": signal_kind.value,
        "input_signal_kind_status": signal_kind.status.value,
        "input_amplitude_units": amplitude_units,
        "input_amplitude_units_status": units_metadata.status.value,
        "filtering_description": filter_metadata.value,
        "filtering_status": filter_metadata.status.value,
        "parameters": parameters.model_dump(mode="json"),
        "threshold": float(result.thresholds[0]),
        "noise_estimate": float(result.noise_estimates[0]),
        "mua_units": "s^-1" if parameters.mode == "rel" else "events s^-1",
        "algorithm_constants": {
            "noise_divisor": NOISE_DIVISOR,
            "prominence_factor": PROMINENCE_FACTOR,
            "amplitude_cap_factor": AMPLITUDE_CAP_FACTOR,
        },
        "dependency_versions": _dependency_versions(),
    }
    artifact_reference = runtime.mua_artifact_store.write(
        mua=result.mua[:, 0],
        times_so=result.times_so[:, 0],
        spike_sample_indices=source_indices,
        spike_times_s=result.spike_times_s[0],
        spike_amplitudes=result.spike_amplitudes[0],
        manifest=manifest,
    )
    output = CalcMUAOutput(
        artifact_reference=artifact_reference,
        dataset_id=reference.dataset.dataset_id,
        source_channel=channel_index,
        sample_interval=reference.sample_interval,
        output_length=int(result.mua.shape[0]),
        event_count=int(relative_indices.size),
        threshold=float(result.thresholds[0]),
        noise_estimate=float(result.noise_estimates[0]),
        stdmin=parameters.stdmin,
        bandwidth_s=parameters.mua_window_s,
        num_points=parameters.num_points,
        mode=parameters.mode,
        mua_units="s^-1" if parameters.mode == "rel" else "events s^-1",
        amplitude_units=amplitude_units,
        amplitude_units_status=units_metadata.status,
        sampling_rate_hz=rate,
        sampling_rate_status=sampling_rate.status,
        filtering_description=(
            str(filter_metadata.value) if filter_metadata.value is not None else None
        ),
        filtering_status=filter_metadata.status,
        source_sha256=SOURCE_SHA256,
        algorithm_constants={
            "noise_divisor": NOISE_DIVISOR,
            "prominence_factor": PROMINENCE_FACTOR,
            "amplitude_cap_factor": AMPLITUDE_CAP_FACTOR,
        },
        dependency_versions=_dependency_versions(),
        execution_id=runtime.execution_id,
    )
    output_metadata = tuple(
        MetadataSnapshot(name=name, value=value.value, status=value.status)
        for name, value in metadata.items()
    )
    warnings = _prepare_warnings(input_data, parameters, context)
    return OperationResult(output=output, output_metadata=output_metadata, warnings=warnings)


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


CALC_MUA_OPERATION = OperationDefinition(
    operation_id=OPERATION_ID,
    name="calcMUA trace",
    description=(
        "NeuroScope Python implementation based on Fiamma Liz Leites' independently "
        "written calcmua_verFi.m implementation of the calcMUA method developed by "
        "Dr. Cecilia Herbert. Applies peak detection and per-channel normal-kernel KDE "
        "to a selected, already-prepared continuous voltage channel; MATLAB numerical "
        "parity has not been established."
    ),
    version=OPERATION_VERSION,
    implementation_version=IMPLEMENTATION_VERSION,
    input_model=CalcMUAInput,
    parameters_model=CalcMUAParameters,
    output_model=CalcMUAOutput,
    execute=_execute_calc_mua,
    metadata_requirements=(
        MetadataRequirement(name="signal_kind", require_confirmed=True),
        MetadataRequirement(name="sampling_rate_hz", require_confirmed=True),
        MetadataRequirement(name="units", required=False),
        MetadataRequirement(name="filtering_description", required=False),
    ),
    warnings=(
        "The algorithm expects caller-supplied filtered_spike_data; no filtering is performed.",
        "Filtering history is recorded as supplied; this operation never applies a filter.",
        "Empty detections produce an all-NaN MUAtrace, matching the MATLAB source.",
        "The source NumPoints value does not control output length because its evaluation grid is explicit.",
        "The uncentered median absolute-amplitude noise estimate can change with baseline offset and event amplitudes.",
        "KDE values near interval edges are not corrected or renormalized by the source method.",
    ),
    limitations=(
        "Negative or positive peak polarity is selected by the sign of stdmin; both are not detected in one run.",
        "No refractory period or event-artifact rejection is applied.",
        "The output is a kernel-density activity trace, not a sorted single-neuron firing rate.",
        "SciPy find_peaks is used as the Python peak backend; MATLAB parity requires reference outputs.",
    ),
    prepare_warnings=_prepare_warnings,
)


def register_calc_mua(registry: OperationRegistry):
    """Register the original method under its stable production operation ID."""
    if not isinstance(registry, OperationRegistry):
        raise TypeError("registry must be an OperationRegistry.")
    registry.register(CALC_MUA_OPERATION)
    return CALC_MUA_OPERATION
