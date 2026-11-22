"""Approved extraction and characterization of detector-aligned waveforms."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
from pydantic import Field, StrictInt, model_validator

from ..artifacts import WaveformArtifactStore
from ..exploration import MetadataStatus, MetadataValue
from .contracts import (
    DataReference,
    DatasetReference,
    MetadataRequirement,
    MetadataSnapshot,
    OperationContext,
    OperationDefinition,
    OperationModel,
    OperationResult,
    OperationRuntimeAccess,
)
from .operations import ExecutionRecord, ExecutionStatus
from .registry import OperationRegistry

OPERATION_ID = "neuroscope.spike.extract-waveforms"
OPERATION_VERSION = "1.0.0"
IMPLEMENTATION_VERSION = "approved-windowed-waveform-extraction-v1"
BOUNDARY_POLICY = "exclude"


class DetectedSpikeEvent(OperationModel):
    """One prior detector event expressed in original recording sample indices."""

    event_id: str = Field(min_length=1)
    channel_index: int = Field(ge=0, strict=True)
    original_sample_index: int = Field(ge=0, strict=True)
    window_relative_sample_index: int | None = Field(default=None, ge=0, strict=True)
    polarity: Literal["positive", "negative"] | None = None
    detector_time_seconds_relative_to_window: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )


class DetectionEventCollection(OperationModel):
    """Serializable handoff of detections; signal samples are never embedded."""

    dataset: DatasetReference
    source_data_reference: DataReference
    detector_operation_id: str = Field(min_length=1)
    detector_operation_version: str = Field(min_length=1)
    detection_reference: str = Field(min_length=1)
    coordinate_system: Literal["original_recording_samples"] = "original_recording_samples"
    sampling_rate_hz: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    sampling_rate_status: MetadataStatus
    amplitude_units: str | None = None
    amplitude_units_status: MetadataStatus
    events: tuple[DetectedSpikeEvent, ...]

    @model_validator(mode="after")
    def validate_event_references(self) -> DetectionEventCollection:
        if self.source_data_reference.dataset != self.dataset:
            raise ValueError("detection events and source DataReference identify different datasets.")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("detected event identifiers must be unique.")
        source = self.source_data_reference
        for event in self.events:
            if event.channel_index not in source.channels:
                raise ValueError("detected event channel is absent from its source DataReference.")
            if not source.sample_interval.start <= event.original_sample_index < source.sample_interval.stop:
                raise ValueError("detected event index is outside its source sample interval.")
            if (
                event.window_relative_sample_index is not None
                and event.original_sample_index
                != source.sample_interval.start + event.window_relative_sample_index
            ):
                raise ValueError("detector window-relative and original sample indices disagree.")
        if self.sampling_rate_status is MetadataStatus.UNKNOWN and self.sampling_rate_hz is not None:
            raise ValueError("unknown sampling rate cannot contain a value.")
        if self.amplitude_units_status is MetadataStatus.UNKNOWN and self.amplitude_units is not None:
            raise ValueError("unknown amplitude units cannot contain a value.")
        return self


class WaveformExtractionInput(OperationModel):
    """Approved recording selection and an immutable prior detection table."""

    signal: DataReference
    detections: DetectionEventCollection

    @model_validator(mode="after")
    def same_dataset_and_registration(self) -> WaveformExtractionInput:
        if self.signal.dataset != self.detections.dataset:
            raise ValueError("waveform signal and detections must refer to the same dataset.")
        if self.signal.session_handle != self.detections.source_data_reference.session_handle:
            raise ValueError("detector events refer to a stale or different dataset registration.")
        if any(channel not in self.signal.channels for channel in self.detections.source_data_reference.channels):
            raise ValueError("waveform DataReference must include all detector input channels.")
        if any(event.channel_index not in self.signal.channels for event in self.detections.events):
            raise ValueError("waveform DataReference omits a detected event channel.")
        return self


class WaveformExtractionParameters(OperationModel):
    """Exact number of original samples before and after each event."""

    pre_samples: int = Field(ge=0, strict=True)
    post_samples: int = Field(ge=0, strict=True)

    @property
    def waveform_length(self) -> int:
        return self.pre_samples + self.post_samples + 1


class ChannelWaveformSummaryReference(OperationModel):
    """Per-channel descriptive summary locations in the waveform artifact."""

    channel_index: int = Field(ge=0, strict=True)
    waveform_count: int = Field(ge=0, strict=True)
    mean_reference: str | None = None
    standard_deviation_reference: str | None = None
    median_reference: str | None = None


class WaveformExtractionOutput(OperationModel):
    """Compact JSON result; individual waveforms and vectors live in the artifact."""

    artifact_reference: str = Field(min_length=1)
    event_mapping_reference: str = Field(min_length=1)
    event_measurements_reference: str = Field(min_length=1)
    excluded_events_reference: str = Field(min_length=1)
    accepted_event_count: int = Field(ge=0, strict=True)
    excluded_event_count: int = Field(ge=0, strict=True)
    waveform_shape: tuple[StrictInt, StrictInt]
    waveform_dtype: str = Field(min_length=1)
    pre_samples: int = Field(ge=0, strict=True)
    post_samples: int = Field(ge=0, strict=True)
    event_sample_offset: int = Field(ge=0, strict=True)
    first_sample_offset_from_event: int
    last_sample_offset_from_event: int
    boundary_policy: Literal["exclude"] = BOUNDARY_POLICY
    detector_operation_id: str
    detector_operation_version: str
    detection_reference: str
    channel_summaries: tuple[ChannelWaveformSummaryReference, ...]
    sampling_rate_hz: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    sampling_rate_status: MetadataStatus
    amplitude_units: str | None = None
    amplitude_units_status: MetadataStatus
    time_reference: Literal["relative_to_recording_sample_zero"] = (
        "relative_to_recording_sample_zero"
    )


class WaveformExtractionError(ValueError):
    """The requested detection table or authorized waveform interval is invalid."""


def detection_collection_from_record(record: ExecutionRecord) -> DetectionEventCollection:
    """Adapt a successful output from either production detector without rerunning it."""
    if not isinstance(record, ExecutionRecord) or record.status is not ExecutionStatus.SUCCEEDED:
        raise ValueError("a successful detector ExecutionRecord is required.")
    operation_id = record.operation_id
    if operation_id not in {
        "neuroscope.spike.detect-negative",
        "neuroscope.spike.detect-signed-peaks",
    }:
        raise ValueError(f"unsupported detector operation {operation_id!r}.")
    input_data = record.input_data
    reference = getattr(input_data, "signal", None)
    output = record.output
    if not isinstance(reference, DataReference) or output is None:
        raise ValueError("detector execution record lacks its signal reference or output.")
    events: list[DetectedSpikeEvent] = []
    sample_indices = output.original_sample_indices
    relative_indices = output.window_relative_sample_indices
    times = output.times_s_relative_to_window
    channel_index = output.channel_index
    polarity = "negative" if operation_id.endswith("detect-negative") else output.polarity
    for row, original_index in enumerate(sample_indices):
        event_polarity = polarity if polarity in {"positive", "negative"} else None
        events.append(
            DetectedSpikeEvent(
                event_id=f"{record.execution_id}:{row}",
                channel_index=channel_index,
                original_sample_index=original_index,
                window_relative_sample_index=relative_indices[row],
                polarity=event_polarity,
                detector_time_seconds_relative_to_window=(
                    None if times is None else times[row]
                ),
            )
        )
    return DetectionEventCollection(
        dataset=reference.dataset,
        source_data_reference=reference,
        detector_operation_id=operation_id,
        detector_operation_version=record.operation_version,
        detection_reference=f"execution://{record.execution_id}",
        sampling_rate_hz=output.sampling_rate_hz,
        sampling_rate_status=output.sampling_rate_status,
        amplitude_units=output.amplitude_units,
        amplitude_units_status=output.amplitude_units_status,
        events=tuple(events),
    )


def _metadata_value(
    name: str,
    actual: MetadataValue,
    requested: dict[str, MetadataSnapshot],
) -> MetadataValue:
    snapshot = requested.get(name)
    if snapshot is None:
        if actual.status is MetadataStatus.UNKNOWN and actual.value is None:
            return actual
        raise WaveformExtractionError(
            f"execution context must snapshot dataset metadata {name!r} and its status."
        )
    if snapshot.status is not actual.status or snapshot.value != actual.value:
        raise WaveformExtractionError(
            f"execution context metadata {name!r} does not match the registered dataset."
        )
    return actual


def _execute_waveform_extraction(
    input_data: WaveformExtractionInput,
    parameters: WaveformExtractionParameters,
    context: OperationContext,
    runtime: OperationRuntimeAccess,
) -> OperationResult[WaveformExtractionOutput]:
    store = runtime.artifact_store
    if not isinstance(store, WaveformArtifactStore):
        raise WaveformExtractionError(
            "waveform extraction requires a WaveformArtifactStore rooted in a persistent project."
        )
    reference = input_data.signal
    resolved = runtime.data_for(reference)
    detections = input_data.detections
    if detections.dataset != reference.dataset:
        raise WaveformExtractionError("detector results belong to a different dataset.")
    if detections.source_data_reference.session_handle != reference.session_handle:
        raise WaveformExtractionError("detector results belong to a stale dataset registration.")
    requested_metadata = {item.name: item for item in context.metadata}
    metadata = {
        name: _metadata_value(name, resolved.metadata[name], requested_metadata)
        for name in ("sampling_rate_hz", "units")
    }
    sampling_rate = metadata["sampling_rate_hz"]
    units = metadata["units"]
    if (
        detections.sampling_rate_status is not sampling_rate.status
        or detections.sampling_rate_hz != sampling_rate.value
        or detections.amplitude_units_status is not units.status
        or detections.amplitude_units != units.value
    ):
        raise WaveformExtractionError(
            "detector output metadata does not match the registered dataset metadata."
        )

    sample_count = resolved.recording_sample_count
    waveform_length = parameters.waveform_length
    accepted: list[tuple[DetectedSpikeEvent, int, int]] = []
    excluded: list[dict[str, object]] = []
    for event in detections.events:
        index = event.original_sample_index
        if not 0 <= index < sample_count:
            raise WaveformExtractionError(
                f"event {event.event_id!r} has an invalid original sample index."
            )
        start = index - parameters.pre_samples
        stop = index + parameters.post_samples + 1
        if start < 0:
            excluded.append(
                {"event_id": event.event_id, "channel_index": event.channel_index,
                 "original_sample_index": index, "reason": "pre_window_outside_recording"}
            )
            continue
        if stop > sample_count:
            excluded.append(
                {"event_id": event.event_id, "channel_index": event.channel_index,
                 "original_sample_index": index, "reason": "post_window_outside_recording"}
            )
            continue
        if (
            start < reference.sample_interval.start
            or stop > reference.sample_interval.stop
        ):
            raise WaveformExtractionError(
                f"event {event.event_id!r} requires samples outside the approved DataReference interval."
            )
        accepted.append((event, start, stop))

    source_dtype = resolved.channel(reference.channels[0]).dtype
    if np.issubdtype(source_dtype, np.complexfloating) or not np.issubdtype(
        source_dtype, np.number
    ):
        raise WaveformExtractionError("waveform extraction requires real numeric signal samples.")

    waveform_count = len(accepted)
    min_values = np.empty(waveform_count, dtype=source_dtype)
    max_values = np.empty(waveform_count, dtype=source_dtype)
    peak_to_peak = np.empty(waveform_count, dtype=np.float64)
    min_positions = np.empty(waveform_count, dtype=np.int64)
    max_positions = np.empty(waveform_count, dtype=np.int64)
    channel_rows: dict[int, list[int]] = defaultdict(list)
    accepted_events: list[dict[str, object]] = []
    rate_confirmed = (
        sampling_rate.status is MetadataStatus.CONFIRMED
        and sampling_rate.value is not None
    )
    confirmed_rate = float(sampling_rate.value) if rate_confirmed else None

    with store.writer(
        shape=(waveform_count, waveform_length),
        dtype=source_dtype,
    ) as writer:
        for row, (event, start, stop) in enumerate(accepted):
            channel_access = resolved.channel(event.channel_index)
            waveform = channel_access.read_window(start, stop)
            if waveform.shape != (waveform_length,) or not np.isfinite(waveform).all():
                raise WaveformExtractionError(
                    f"event {event.event_id!r} waveform is malformed or non-finite."
                )
            writer.write_row(row, waveform)
            min_position = int(np.argmin(waveform))
            max_position = int(np.argmax(waveform))
            min_values[row] = waveform[min_position]
            max_values[row] = waveform[max_position]
            peak_to_peak[row] = float(waveform[max_position]) - float(waveform[min_position])
            min_positions[row] = min_position - parameters.pre_samples
            max_positions[row] = max_position - parameters.pre_samples
            channel_rows[event.channel_index].append(row)
            accepted_events.append(
                {
                    "event_id": event.event_id,
                    "waveform_row": row,
                    "channel_index": event.channel_index,
                    "original_sample_index": event.original_sample_index,
                    "detector_window_relative_sample_index": event.window_relative_sample_index,
                    "polarity": event.polarity,
                    "detector_time_seconds_relative_to_window": (
                        event.detector_time_seconds_relative_to_window
                    ),
                    "sample_time_seconds_relative_to_recording_sample_zero": (
                        event.original_sample_index / confirmed_rate
                        if confirmed_rate is not None
                        else None
                    ),
                    "waveform_sample_start": start,
                    "waveform_sample_stop_exclusive": stop,
                }
            )

        measurements = {
            "minimum_amplitude": min_values,
            "maximum_amplitude": max_values,
            "peak_to_peak_amplitude": peak_to_peak,
            "minimum_position_from_event": min_positions,
            "maximum_position_from_event": max_positions,
        }
        writer.write_measurements(measurements)
        channel_summary_descriptors: list[ChannelWaveformSummaryReference] = []
        for channel in reference.channels:
            row_indices = channel_rows.get(channel, [])
            if row_indices:
                mean = np.empty(waveform_length, dtype=np.float64)
                standard_deviation = np.empty(waveform_length, dtype=np.float64)
                median = np.empty(waveform_length, dtype=np.float64)
                for offset in range(waveform_length):
                    column = np.asarray(writer.waveforms[row_indices, offset], dtype=np.float64)
                    mean[offset] = np.mean(column)
                    standard_deviation[offset] = np.std(column, ddof=0)
                    median[offset] = np.median(column)
                filenames = writer.write_channel_summary(
                    channel,
                    mean=mean,
                    standard_deviation=standard_deviation,
                    median=median,
                )
                summary = ChannelWaveformSummaryReference(
                    channel_index=channel,
                    waveform_count=len(row_indices),
                    mean_reference=filenames["mean"],
                    standard_deviation_reference=(
                        filenames["standard-deviation"]
                    ),
                    median_reference=filenames["median"],
                )
            else:
                summary = ChannelWaveformSummaryReference(
                    channel_index=channel,
                    waveform_count=0,
                )
            channel_summary_descriptors.append(summary)

        manifest = {
            "schema_version": "1.0.0",
            "operation_id": OPERATION_ID,
            "operation_version": OPERATION_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "execution_id": runtime.execution_id,
            "execution_started_at_utc": runtime.started_at_utc.isoformat(),
            "dataset_reference": reference.dataset.model_dump(mode="json"),
            "data_reference": reference.model_dump(mode="json"),
            "detector_operation_id": detections.detector_operation_id,
            "detector_operation_version": detections.detector_operation_version,
            "detection_reference": detections.detection_reference,
            "detector_data_reference": detections.source_data_reference.model_dump(mode="json"),
            "parameters": parameters.model_dump(mode="json"),
            "boundary_policy": BOUNDARY_POLICY,
            "waveform_shape": [waveform_count, waveform_length],
            "waveform_dtype": str(source_dtype),
            "sample_offsets": {
                "start_inclusive": -parameters.pre_samples,
                "stop_exclusive": parameters.post_samples + 1,
                "event_offset": parameters.pre_samples,
            },
            "sampling_rate_hz": sampling_rate.value,
            "sampling_rate_status": sampling_rate.status.value,
            "amplitude_units": units.value,
            "amplitude_units_status": units.status.value,
            "channel_summaries": [item.model_dump(mode="json") for item in channel_summary_descriptors],
            "accepted_events": accepted_events,
            "excluded_events": excluded,
        }
        artifact_reference = writer.commit(manifest)

    output = WaveformExtractionOutput(
        artifact_reference=artifact_reference,
        event_mapping_reference=f"{artifact_reference}#accepted-events",
        event_measurements_reference=f"{artifact_reference}#measurements",
        excluded_events_reference=f"{artifact_reference}#excluded-events",
        accepted_event_count=waveform_count,
        excluded_event_count=len(excluded),
        waveform_shape=(waveform_count, waveform_length),
        waveform_dtype=str(source_dtype),
        pre_samples=parameters.pre_samples,
        post_samples=parameters.post_samples,
        event_sample_offset=parameters.pre_samples,
        first_sample_offset_from_event=-parameters.pre_samples,
        last_sample_offset_from_event=parameters.post_samples,
        detector_operation_id=detections.detector_operation_id,
        detector_operation_version=detections.detector_operation_version,
        detection_reference=detections.detection_reference,
        channel_summaries=tuple(channel_summary_descriptors),
        sampling_rate_hz=(float(sampling_rate.value) if sampling_rate.value is not None else None),
        sampling_rate_status=sampling_rate.status,
        amplitude_units=str(units.value) if units.value is not None else None,
        amplitude_units_status=units.status,
    )
    output_metadata = tuple(
        MetadataSnapshot(name=name, value=value.value, status=value.status)
        for name, value in metadata.items()
    )
    return OperationResult(output, output_metadata)


WAVEFORM_EXTRACTION = OperationDefinition(
    operation_id=OPERATION_ID,
    name="Spike waveform extraction and characterization",
    description=(
        "Extract detector-aligned fixed sample windows around previously detected events "
        "and preserve individual waveforms with per-channel descriptive summaries."
    ),
    version=OPERATION_VERSION,
    implementation_version=IMPLEMENTATION_VERSION,
    input_model=WaveformExtractionInput,
    parameters_model=WaveformExtractionParameters,
    output_model=WaveformExtractionOutput,
    execute=_execute_waveform_extraction,
    metadata_requirements=(
        MetadataRequirement(name="sampling_rate_hz", required=False),
        MetadataRequirement(name="units", required=False),
    ),
    warnings=(
        "Waveforms are aligned to original detector sample indices; no realignment is applied.",
        "Only descriptive waveform statistics are calculated; shape alone does not establish neuron identity.",
        "Synthetic validation does not establish biological validity.",
    ),
    limitations=(
        "Boundary events with incomplete requested windows are excluded without truncation or padding.",
        "Mean, standard deviation (population ddof=0), and median summaries are calculated per channel.",
        "No preprocessing, baseline subtraction, normalization, interpolation, or sub-sample alignment is performed.",
    ),
)


def register_waveform_extraction(
    registry: OperationRegistry,
) -> OperationDefinition[
    WaveformExtractionInput,
    WaveformExtractionParameters,
    WaveformExtractionOutput,
]:
    """Register the waveform operation in the existing operation registry."""
    if not isinstance(registry, OperationRegistry):
        raise TypeError("registry must be an OperationRegistry.")
    registry.register(WAVEFORM_EXTRACTION)
    return WAVEFORM_EXTRACTION
