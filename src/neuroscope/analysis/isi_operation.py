"""Approved, detector-independent inter-spike interval summaries."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from math import isfinite
from numbers import Real
from typing import Literal

import numpy as np
from pydantic import Field, StrictInt, field_validator, model_validator

from ..artifacts import ISIArtifactStore
from ..exploration import MetadataStatus
from .contracts import (
    DatasetReference,
    MetadataRequirement,
    MetadataSnapshot,
    OperationContext,
    OperationDefinition,
    OperationModel,
    OperationResult,
    OperationRuntimeAccess,
    SampleInterval,
)
from .operations import ExecutionRecord
from .registry import OperationRegistry
from .waveform_operation import (
    DetectedSpikeEvent,
    DetectionEventCollection,
)

OPERATION_ID = "neuroscope.spike.analyze-isi"
OPERATION_VERSION = "1.0.0"
IMPLEMENTATION_VERSION = "event-group-sort-streaming-isi-v1"


class ISIEventCollection(OperationModel):
    """Event-only detector handoff; it deliberately contains no recording handle."""

    dataset: DatasetReference
    source_sample_interval: SampleInterval
    source_channels: tuple[StrictInt, ...] = Field(min_length=1)
    detector_operation_id: Literal[
        "neuroscope.spike.detect-negative",
        "neuroscope.spike.detect-signed-peaks",
    ]
    detector_operation_version: str = Field(min_length=1)
    detection_reference: str = Field(min_length=1)
    detector_execution_id: str | None = None
    sampling_rate_hz: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    sampling_rate_status: MetadataStatus
    events: tuple[DetectedSpikeEvent, ...]

    @model_validator(mode="after")
    def validate_events(self) -> ISIEventCollection:
        if len(set(self.source_channels)) != len(self.source_channels):
            raise ValueError("source channel indices must be unique.")
        if any(channel < 0 for channel in self.source_channels):
            raise ValueError("source channel indices must be non-negative.")
        event_ids = [event.event_id for event in self.events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("event identifiers must be unique.")
        for event in self.events:
            if event.channel_index not in self.source_channels:
                raise ValueError("event channel is absent from the detector selection.")
            if not (
                self.source_sample_interval.start
                <= event.original_sample_index
                < self.source_sample_interval.stop
            ):
                raise ValueError("event index is outside the detector sample interval.")
            if (
                event.window_relative_sample_index is not None
                and event.original_sample_index
                != self.source_sample_interval.start + event.window_relative_sample_index
            ):
                raise ValueError("event sample coordinates do not agree.")
        if self.sampling_rate_status is MetadataStatus.UNKNOWN and self.sampling_rate_hz is not None:
            raise ValueError("unknown sampling rate cannot contain a value.")
        if self.detection_reference.startswith("execution://"):
            referenced_id = self.detection_reference.removeprefix("execution://")
            if not referenced_id:
                raise ValueError("execution detection reference must contain an execution ID.")
            if self.detector_execution_id not in (None, referenced_id):
                raise ValueError("detector execution ID does not match its execution reference.")
            if self.detector_execution_id is None:
                object.__setattr__(self, "detector_execution_id", referenced_id)
        return self


class ISIInput(OperationModel):
    """Previously detected events and their dataset/detector provenance."""

    events: ISIEventCollection


class ISIParameters(OperationModel):
    """Explicit polarity grouping and optional short-interval threshold."""

    polarity_mode: Literal["separate", "combine"]
    short_interval_threshold_samples: int | None = Field(default=None, ge=0, strict=True)
    short_interval_threshold_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator("short_interval_threshold_ms", mode="before")
    @classmethod
    def finite_millisecond_threshold(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("short_interval_threshold_ms must be a finite real number.")
        if not isfinite(float(value)):
            raise ValueError("short_interval_threshold_ms must be finite.")
        return float(value)

    @model_validator(mode="after")
    def threshold_units_are_unambiguous(self) -> ISIParameters:
        if (
            self.short_interval_threshold_samples is not None
            and self.short_interval_threshold_ms is not None
        ):
            raise ValueError("provide a short-interval threshold in samples or milliseconds, not both.")
        return self


class ISIInterval(OperationModel):
    """One consecutive within-channel interval in original recording samples."""

    starting_event_id: str
    ending_event_id: str
    channel_index: int = Field(ge=0, strict=True)
    polarity_group: Literal["positive", "negative", "unknown", "combined"]
    starting_sample_index: int = Field(ge=0, strict=True)
    ending_sample_index: int = Field(ge=0, strict=True)
    delta_samples: int = Field(ge=0, strict=True)
    delta_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def interval_matches_coordinates(self) -> ISIInterval:
        if self.ending_sample_index - self.starting_sample_index != self.delta_samples:
            raise ValueError("delta_samples must equal the difference between event indices.")
        return self


class ISIGroupSummary(OperationModel):
    """Descriptive interval statistics for one channel and polarity group."""

    channel_index: int = Field(ge=0, strict=True)
    polarity_group: Literal["positive", "negative", "unknown", "combined"]
    detected_event_count: int = Field(ge=0, strict=True)
    interval_count: int = Field(ge=0, strict=True)
    duplicate_event_count: int = Field(ge=0, strict=True)
    zero_length_interval_count: int = Field(ge=0, strict=True)
    mean_isi_samples: float | None = None
    median_isi_samples: float | None = None
    standard_deviation_isi_samples: float | None = None
    minimum_isi_samples: int | None = None
    maximum_isi_samples: int | None = None
    mean_isi_ms: float | None = None
    median_isi_ms: float | None = None
    standard_deviation_isi_ms: float | None = None
    minimum_isi_ms: float | None = None
    maximum_isi_ms: float | None = None
    coefficient_of_variation: float | None = None
    intervals_below_threshold_count: int | None = None
    short_interval_fraction: float | None = None


class ISIOutput(OperationModel):
    """Compact summaries plus a project artifact reference for individual intervals."""

    dataset: DatasetReference
    detector_operation_id: str
    detector_operation_version: str
    detection_reference: str
    detector_execution_id: str | None = None
    polarity_mode: Literal["separate", "combine"]
    event_count: int = Field(ge=0, strict=True)
    interval_count: int = Field(ge=0, strict=True)
    duplicate_event_count: int = Field(ge=0, strict=True)
    zero_length_interval_count: int = Field(ge=0, strict=True)
    interval_artifact_reference: str = Field(min_length=1)
    summaries: tuple[ISIGroupSummary, ...]
    short_interval_threshold_samples: int | None = None
    short_interval_threshold_ms: float | None = None
    sampling_rate_hz: float | None = None
    sampling_rate_status: MetadataStatus
    interval_time_unit: Literal["milliseconds", "unavailable"]
    standard_deviation_convention: Literal["population-ddof-0"] = "population-ddof-0"


def isi_collection_from_detection_collection(
    collection: DetectionEventCollection,
) -> ISIEventCollection:
    """Drop only the signal handle from the waveform-shared detector handoff."""
    if not isinstance(collection, DetectionEventCollection):
        raise TypeError("collection must be a DetectionEventCollection.")
    return ISIEventCollection(
        dataset=collection.dataset,
        source_sample_interval=collection.source_data_reference.sample_interval,
        source_channels=collection.source_data_reference.channels,
        detector_operation_id=collection.detector_operation_id,
        detector_operation_version=collection.detector_operation_version,
        detection_reference=collection.detection_reference,
        detector_execution_id=(
            collection.detection_reference.removeprefix("execution://")
            if collection.detection_reference.startswith("execution://")
            else None
        ),
        sampling_rate_hz=collection.sampling_rate_hz,
        sampling_rate_status=collection.sampling_rate_status,
        events=collection.events,
    )


def isi_collection_from_record(record: ExecutionRecord) -> ISIEventCollection:
    """Adapt a successful production detector execution without reading signal data."""
    from .waveform_operation import detection_collection_from_record

    return isi_collection_from_detection_collection(detection_collection_from_record(record))


def _group_key(event: DetectedSpikeEvent, polarity_mode: str) -> str:
    if polarity_mode == "combine":
        return "combined"
    return event.polarity or "unknown"


def _prepare_isi_warnings(
    input_data: ISIInput,
    parameters: ISIParameters,
    context: OperationContext,
) -> tuple[str, ...]:
    event_collection = input_data.events
    if context.dataset_reference is not None and context.dataset_reference != event_collection.dataset:
        raise ValueError("operation context dataset identity does not match the event collection.")
    if context.selected_channels and context.selected_channels != event_collection.source_channels:
        raise ValueError("operation context channels do not match the detector event selection.")
    if (
        context.sample_interval is not None
        and context.sample_interval != event_collection.source_sample_interval
    ):
        raise ValueError("operation context interval does not match the detector event selection.")
    rate_snapshot = next(
        (item for item in context.metadata if item.name == "sampling_rate_hz"),
        None,
    )
    if rate_snapshot is not None and (
        rate_snapshot.value != event_collection.sampling_rate_hz
        or rate_snapshot.status is not event_collection.sampling_rate_status
    ):
        raise ValueError("sampling-rate context does not match detector provenance.")
    if parameters.short_interval_threshold_ms is not None and (
        event_collection.sampling_rate_status is not MetadataStatus.CONFIRMED
        or event_collection.sampling_rate_hz is None
    ):
        raise ValueError(
            "a threshold in milliseconds requires a confirmed sampling rate."
        )
    groups: dict[tuple[int, str], list[DetectedSpikeEvent]] = defaultdict(list)
    for event in event_collection.events:
        groups[(event.channel_index, _group_key(event, parameters.polarity_mode))].append(event)
    duplicate_present = any(
        len(items) != len({event.original_sample_index for event in items})
        for items in groups.values()
    )
    warnings: list[str] = []
    if event_collection.sampling_rate_status is MetadataStatus.UNKNOWN:
        warnings.append("Sampling rate is unknown; only sample-based intervals are available.")
    elif event_collection.sampling_rate_status is MetadataStatus.INFERRED:
        warnings.append("Sampling rate is inferred and unconfirmed; time conversion is unavailable.")
    if duplicate_present:
        warnings.append(
            "Duplicate event indices produce zero-length intervals and may reflect merged event sets or repeated detections; review them, and do not interpret duplicates as biological events."
        )
    if parameters.short_interval_threshold_samples is None and parameters.short_interval_threshold_ms is None:
        warnings.append("No short-interval threshold was supplied; threshold-dependent metrics are unavailable.")
    return tuple(warnings)


def _summary_for_group(
    channel_index: int,
    polarity_group: str,
    events: list[DetectedSpikeEvent],
    deltas: list[int],
    *,
    sampling_rate_hz: float | None,
    threshold_samples: float | None,
) -> ISIGroupSummary:
    interval_count = len(deltas)
    duplicates = len(events) - len({item.original_sample_index for item in events})
    zero_count = sum(delta == 0 for delta in deltas)
    if deltas:
        values = np.asarray(deltas, dtype=np.float64)
        mean = float(np.mean(values))
        median = float(np.median(values))
        standard_deviation = float(np.std(values, ddof=0))
        minimum = min(deltas)
        maximum = max(deltas)
        cv = standard_deviation / mean if mean != 0 else None
        below_count = (
            sum(delta < threshold_samples for delta in deltas)
            if threshold_samples is not None
            else None
        )
        fraction = below_count / interval_count if below_count is not None else None
        if sampling_rate_hz is not None:
            scale = 1000.0 / sampling_rate_hz
            values_ms = values * scale
            mean_ms = float(np.mean(values_ms))
            median_ms = float(np.median(values_ms))
            standard_deviation_ms = float(np.std(values_ms, ddof=0))
            minimum_ms = minimum * scale
            maximum_ms = maximum * scale
        else:
            mean_ms = median_ms = standard_deviation_ms = minimum_ms = maximum_ms = None
    else:
        mean = median = standard_deviation = None
        minimum = maximum = cv = None
        below_count = 0 if threshold_samples is not None else None
        fraction = None
        mean_ms = median_ms = standard_deviation_ms = minimum_ms = maximum_ms = None
    return ISIGroupSummary(
        channel_index=channel_index,
        polarity_group=polarity_group,
        detected_event_count=len(events),
        interval_count=interval_count,
        duplicate_event_count=duplicates,
        zero_length_interval_count=zero_count,
        mean_isi_samples=mean,
        median_isi_samples=median,
        standard_deviation_isi_samples=standard_deviation,
        minimum_isi_samples=minimum,
        maximum_isi_samples=maximum,
        mean_isi_ms=mean_ms,
        median_isi_ms=median_ms,
        standard_deviation_isi_ms=standard_deviation_ms,
        minimum_isi_ms=minimum_ms,
        maximum_isi_ms=maximum_ms,
        coefficient_of_variation=cv,
        intervals_below_threshold_count=below_count,
        short_interval_fraction=fraction,
    )


def _execute_isi(
    input_data: ISIInput,
    parameters: ISIParameters,
    context: OperationContext,
    runtime: OperationRuntimeAccess,
) -> OperationResult[ISIOutput]:
    del context
    store = runtime.isi_artifact_store
    if not isinstance(store, ISIArtifactStore):
        raise TypeError("ISI analysis requires the current project's ISIArtifactStore.")
    collection = input_data.events
    confirmed_rate = (
        float(collection.sampling_rate_hz)
        if collection.sampling_rate_status is MetadataStatus.CONFIRMED
        and collection.sampling_rate_hz is not None
        else None
    )
    if parameters.short_interval_threshold_ms is not None:
        threshold_samples = (
            parameters.short_interval_threshold_ms * confirmed_rate / 1000.0
        )
    elif parameters.short_interval_threshold_samples is not None:
        threshold_samples = float(parameters.short_interval_threshold_samples)
    else:
        threshold_samples = None

    grouped: dict[tuple[int, str], list[DetectedSpikeEvent]] = defaultdict(list)
    for event in collection.events:
        grouped[(event.channel_index, _group_key(event, parameters.polarity_mode))].append(event)
    ordered_keys = sorted(grouped, key=lambda key: (key[0], key[1]))
    if not ordered_keys and parameters.polarity_mode == "combine":
        ordered_keys = [(channel, "combined") for channel in collection.source_channels]
    elif not ordered_keys:
        ordered_keys = [(channel, "unknown") for channel in collection.source_channels]
    interval_count = duplicate_count = zero_count = 0
    summaries: list[ISIGroupSummary] = []
    warnings = list(_prepare_isi_warnings(input_data, parameters, OperationContext()))
    started = runtime.started_at_utc.isoformat()
    with store.writer() as writer:
        for channel_index, polarity_group in ordered_keys:
            events = sorted(
                grouped[(channel_index, polarity_group)],
                key=lambda event: (event.original_sample_index, event.event_id),
            )
            deltas: list[int] = []
            for first, second in pairwise(events):
                delta = second.original_sample_index - first.original_sample_index
                delta_ms = (
                    1000.0 * delta / confirmed_rate if confirmed_rate is not None else None
                )
                interval = ISIInterval(
                    starting_event_id=first.event_id,
                    ending_event_id=second.event_id,
                    channel_index=channel_index,
                    polarity_group=polarity_group,
                    starting_sample_index=first.original_sample_index,
                    ending_sample_index=second.original_sample_index,
                    delta_samples=delta,
                    delta_ms=delta_ms,
                )
                writer.write_row(interval.model_dump(mode="json"))
                deltas.append(delta)
            summary = _summary_for_group(
                channel_index,
                polarity_group,
                events,
                deltas,
                sampling_rate_hz=confirmed_rate,
                threshold_samples=threshold_samples,
            )
            summaries.append(summary)
            interval_count += summary.interval_count
            duplicate_count += summary.duplicate_event_count
            zero_count += summary.zero_length_interval_count

        manifest = {
            "schema_version": "1.0.0",
            "operation_id": OPERATION_ID,
            "operation_version": OPERATION_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "execution_id": runtime.execution_id,
            "execution_started_at_utc": started,
            "dataset_reference": collection.dataset.model_dump(mode="json"),
            "detector_operation_id": collection.detector_operation_id,
            "detector_operation_version": collection.detector_operation_version,
            "detection_reference": collection.detection_reference,
            "detector_execution_id": collection.detector_execution_id,
            "event_count": len(collection.events),
            "source_sample_interval": collection.source_sample_interval.model_dump(mode="json"),
            "source_channels": list(collection.source_channels),
            "polarity_mode": parameters.polarity_mode,
            "threshold_samples": parameters.short_interval_threshold_samples,
            "threshold_ms": parameters.short_interval_threshold_ms,
            "sampling_rate_hz": collection.sampling_rate_hz,
            "sampling_rate_status": collection.sampling_rate_status.value,
            "interval_count": interval_count,
            "duplicate_event_count": duplicate_count,
            "zero_length_interval_count": zero_count,
            "warnings": warnings,
            "summaries": [item.model_dump(mode="json") for item in summaries],
        }
        artifact_reference = writer.commit(manifest)

    output = ISIOutput(
        dataset=collection.dataset,
        detector_operation_id=collection.detector_operation_id,
        detector_operation_version=collection.detector_operation_version,
        detection_reference=collection.detection_reference,
        detector_execution_id=collection.detector_execution_id,
        polarity_mode=parameters.polarity_mode,
        event_count=len(collection.events),
        interval_count=interval_count,
        duplicate_event_count=duplicate_count,
        zero_length_interval_count=zero_count,
        interval_artifact_reference=artifact_reference,
        summaries=tuple(summaries),
        short_interval_threshold_samples=parameters.short_interval_threshold_samples,
        short_interval_threshold_ms=parameters.short_interval_threshold_ms,
        sampling_rate_hz=collection.sampling_rate_hz,
        sampling_rate_status=collection.sampling_rate_status,
        interval_time_unit="milliseconds" if confirmed_rate is not None else "unavailable",
    )
    output_metadata = (
        MetadataSnapshot(
            name="sampling_rate_hz",
            value=collection.sampling_rate_hz,
            status=collection.sampling_rate_status,
        ),
    )
    return OperationResult(output, output_metadata, tuple(warnings))


ISI_ANALYSIS = OperationDefinition(
    operation_id=OPERATION_ID,
    name="Inter-spike interval analysis",
    description=(
        "Calculate per-channel consecutive detected-event intervals and descriptive "
        "statistics without accessing the recording signal."
    ),
    version=OPERATION_VERSION,
    implementation_version=IMPLEMENTATION_VERSION,
    input_model=ISIInput,
    parameters_model=ISIParameters,
    output_model=ISIOutput,
    execute=_execute_isi,
    metadata_requirements=(MetadataRequirement(name="sampling_rate_hz", required=False),),
    warnings=(
        "Detected events are not necessarily sorted single-neuron units.",
        "Short-interval fractions are descriptive and do not establish refractory violations.",
    ),
    limitations=(
        "ISI is computed independently within each channel and selected polarity group.",
        "Population standard deviation with ddof=0 is used.",
        "No events are rejected or classified based on interval length.",
    ),
    prepare_warnings=_prepare_isi_warnings,
)


def register_isi_analysis(registry: OperationRegistry) -> OperationDefinition[ISIInput, ISIParameters, ISIOutput]:
    """Register the ISI analysis definition in the existing operation registry."""
    if not isinstance(registry, OperationRegistry):
        raise TypeError("registry must be an OperationRegistry.")
    registry.register(ISI_ANALYSIS)
    return ISI_ANALYSIS
