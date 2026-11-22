"""Deterministic synthetic ground-truth evaluation for negative spike detection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import fsum
from statistics import median
from typing import Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from ..data.synthetic import (
    GroundTruthSpike,
    SyntheticRecordingConfig,
    generate_synthetic_recording,
)
from ..data_access import DatasetRegistration, DatasetStore
from ..domain import DatasetProfile
from ..exploration import DatasetExploration, MetadataStatus
from .contracts import (
    DatasetReference,
    ExecutionStatus,
    MetadataSnapshot,
    OperationContext,
    OperationModel,
    ReproducibilityInfo,
    SampleInterval,
)
from .operations import (
    ApprovedExecution,
    ExecutionRecord,
    OperationRunner,
    PreparedExecution,
)
from .registry import OperationRegistry
from .signed_peak_operation import (
    OPERATION_ID as SIGNED_PEAK_OPERATION_ID,
)
from .signed_peak_operation import (
    OPERATION_VERSION as SIGNED_PEAK_OPERATION_VERSION,
)
from .signed_peak_operation import (
    SIGNED_PEAK_DETECTOR,
    SignedPeakInput,
    SignedPeakOutput,
    SignedPeakParameters,
    SignedPeakProvenance,
)
from .spike_operation import (
    NEGATIVE_SPIKE_DETECTOR,
    OPERATION_ID,
    OPERATION_VERSION,
    NegativeSpikeInput,
    NegativeSpikeOutput,
    NegativeSpikeParameters,
)

MATCHING_ALGORITHM_ID = "ordered-dynamic-programming-cardinality-error"
MATCHING_ALGORITHM_VERSION = "1"


class DetectionEvent(OperationModel):
    """One detector event with explicit window and original sample coordinates."""

    detection_index: int = Field(ge=0, strict=True)
    channel_index: int = Field(ge=0, strict=True)
    window_relative_sample_index: int = Field(ge=0, strict=True)
    original_sample_index: int = Field(ge=0, strict=True)


class MatchedEvent(OperationModel):
    """One one-to-one detected/ground-truth pair and its signed timing error."""

    channel_index: int = Field(ge=0, strict=True)
    ground_truth_event_id: str = Field(min_length=1)
    ground_truth_sample_index: int = Field(ge=0, strict=True)
    detection_index: int = Field(ge=0, strict=True)
    detected_window_relative_sample_index: int = Field(ge=0, strict=True)
    detected_original_sample_index: int = Field(ge=0, strict=True)
    signed_timing_error_samples: int
    absolute_timing_error_samples: int = Field(ge=0)
    signed_timing_error_seconds: float | None = Field(default=None, allow_inf_nan=False)
    absolute_timing_error_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class UnmatchedGroundTruthEvent(OperationModel):
    event_id: str = Field(min_length=1)
    channel_index: int = Field(ge=0, strict=True)
    sample_index: int = Field(ge=0, strict=True)


class ExcludedPositiveEvent(OperationModel):
    """Positive-going ground truth, reported but excluded from negative-detector FN."""

    event_id: str = Field(min_length=1)
    channel_index: int = Field(ge=0, strict=True)
    sample_index: int = Field(ge=0, strict=True)


class EvaluationMetrics(OperationModel):
    """Counts and timing summaries; undefined ratios/statistics are None."""

    true_positives: int = Field(ge=0, strict=True)
    false_positives: int = Field(ge=0, strict=True)
    false_negatives: int = Field(ge=0, strict=True)
    precision: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    recall_sensitivity: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    f1_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    mean_absolute_timing_error_samples: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    median_absolute_timing_error_samples: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    maximum_absolute_timing_error_samples: int | None = Field(default=None, ge=0, strict=True)
    mean_signed_timing_error_samples: float | None = Field(default=None, allow_inf_nan=False)
    median_signed_timing_error_samples: float | None = Field(default=None, allow_inf_nan=False)
    mean_absolute_timing_error_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    median_absolute_timing_error_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    maximum_absolute_timing_error_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    mean_signed_timing_error_seconds: float | None = Field(default=None, allow_inf_nan=False)
    median_signed_timing_error_seconds: float | None = Field(default=None, allow_inf_nan=False)


class ChannelEvaluation(OperationModel):
    channel_index: int = Field(ge=0, strict=True)
    negative_target_event_count: int = Field(ge=0, strict=True)
    positive_excluded_event_count: int = Field(ge=0, strict=True)
    detection_count: int = Field(ge=0, strict=True)
    metrics: EvaluationMetrics
    matched_events: tuple[MatchedEvent, ...]
    unmatched_detections: tuple[DetectionEvent, ...]
    unmatched_ground_truth_events: tuple[UnmatchedGroundTruthEvent, ...]
    excluded_positive_events: tuple[ExcludedPositiveEvent, ...]


class EventEvaluation(OperationModel):
    """Evaluation restricted to one original-recording half-open sample interval."""

    sample_interval: SampleInterval
    matching_tolerance_samples: int = Field(ge=0, strict=True)
    matching_algorithm_id: str
    matching_algorithm_version: str
    total_negative_target_events: int = Field(ge=0, strict=True)
    total_positive_excluded_events: int = Field(ge=0, strict=True)
    total_detections: int = Field(ge=0, strict=True)
    number_of_matches: int = Field(ge=0, strict=True)
    number_of_unmatched_detections: int = Field(ge=0, strict=True)
    number_of_unmatched_ground_truth_events: int = Field(ge=0, strict=True)
    per_channel: tuple[ChannelEvaluation, ...]
    aggregate_micro_metrics: EvaluationMetrics
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class ExcludedGroundTruthEvent(OperationModel):
    event_id: str = Field(min_length=1)
    channel_index: int = Field(ge=0, strict=True)
    sample_index: int = Field(ge=0, strict=True)
    polarity: Literal["negative", "positive"]


class PolarityChannelEvaluation(OperationModel):
    channel_index: int = Field(ge=0, strict=True)
    target_event_count: int = Field(ge=0, strict=True)
    opposite_polarity_excluded_event_count: int = Field(ge=0, strict=True)
    detection_count: int = Field(ge=0, strict=True)
    metrics: EvaluationMetrics
    matched_events: tuple[MatchedEvent, ...]
    unmatched_detections: tuple[DetectionEvent, ...]
    unmatched_ground_truth_events: tuple[UnmatchedGroundTruthEvent, ...]
    excluded_ground_truth_events: tuple[ExcludedGroundTruthEvent, ...]


class PolarityEventEvaluation(OperationModel):
    target_polarity: Literal["negative", "positive"]
    sample_interval: SampleInterval
    matching_tolerance_samples: int = Field(ge=0, strict=True)
    matching_algorithm_id: str
    matching_algorithm_version: str
    total_target_events: int = Field(ge=0, strict=True)
    total_opposite_polarity_excluded_events: int = Field(ge=0, strict=True)
    total_detections: int = Field(ge=0, strict=True)
    number_of_matches: int = Field(ge=0, strict=True)
    number_of_unmatched_detections: int = Field(ge=0, strict=True)
    number_of_unmatched_ground_truth_events: int = Field(ge=0, strict=True)
    per_channel: tuple[PolarityChannelEvaluation, ...]
    aggregate_micro_metrics: EvaluationMetrics
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class ChannelDetectionSummary(OperationModel):
    channel_index: int = Field(ge=0, strict=True)
    threshold_mode: str
    polarity: Literal["negative", "positive", "none"]
    noise_estimate: float = Field(ge=0, allow_inf_nan=False)
    noise_source: str
    threshold: float = Field(allow_inf_nan=False)
    detection_count: int = Field(ge=0, strict=True)


class DetectorRunEvaluation(OperationModel):
    run_id: str
    operation_id: str
    operation_version: str
    implementation_version: str
    parameter_snapshot: NegativeSpikeParameters | SignedPeakParameters
    polarity_evaluation: PolarityEventEvaluation
    channel_summaries: tuple[ChannelDetectionSummary, ...]
    execution_ids: tuple[str, ...]
    algorithm_provenance: SignedPeakProvenance | None = None
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


class SignedPeakRunConfiguration(OperationModel):
    run_id: str = Field(min_length=1)
    target_polarity: Literal["negative", "positive"]
    parameters: SignedPeakParameters

    @model_validator(mode="after")
    def polarity_matches_threshold(self) -> SignedPeakRunConfiguration:
        if self.parameters.polarity != "none" and self.parameters.polarity != self.target_polarity:
            raise ValueError("target_polarity must match the signed detector threshold.")
        return self


class SyntheticDetectorComparisonScenario(OperationModel):
    scenario_id: str = Field(min_length=1)
    generator_config: SyntheticRecordingConfig
    negative_detector_parameters: NegativeSpikeParameters
    signed_peak_runs: tuple[SignedPeakRunConfiguration, ...] = Field(min_length=1)
    matching_tolerance_samples: int = Field(ge=0, strict=True)
    baseline_offset_by_channel: tuple[float, ...] | None = None
    filtering_description: str | None = None
    filtering_status: MetadataStatus = MetadataStatus.UNKNOWN

    @field_validator("baseline_offset_by_channel")
    @classmethod
    def finite_baseline_offsets(cls, value: tuple[float, ...] | None):
        if value is not None and (
            not value or not np.isfinite(np.asarray(value, dtype=float)).all()
        ):
            raise ValueError("baseline offsets must be a non-empty finite sequence.")
        return value

    @model_validator(mode="after")
    def scenario_metadata_is_consistent(self) -> SyntheticDetectorComparisonScenario:
        if self.baseline_offset_by_channel is not None and len(self.baseline_offset_by_channel) != self.generator_config.channel_count:
            raise ValueError("baseline_offset_by_channel must have one value per channel.")
        if self.filtering_status is not MetadataStatus.UNKNOWN and (
            self.filtering_description is None
            or not self.filtering_description.strip()
        ):
            raise ValueError("non-unknown filtering metadata requires a description.")
        run_ids = [run.run_id for run in self.signed_peak_runs]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("signed peak run IDs must be unique.")
        return self


class SyntheticDetectorComparisonResult(OperationModel):
    scenario_id: str
    generator_config: SyntheticRecordingConfig
    random_seed: int = Field(ge=0, strict=True)
    baseline_offset_by_channel: tuple[float, ...] | None
    dataset_reference: DatasetReference
    metadata_snapshot: tuple[MetadataSnapshot, ...]
    matching_tolerance_samples: int = Field(ge=0, strict=True)
    negative_detector: DetectorRunEvaluation
    signed_peak_runs: tuple[DetectorRunEvaluation, ...]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


class SyntheticEvaluationScenario(OperationModel):
    """One explicit generator/detector/matching configuration; no scientific defaults."""

    scenario_id: str = Field(min_length=1)
    generator_config: SyntheticRecordingConfig
    detector_parameters: NegativeSpikeParameters
    matching_tolerance_samples: int = Field(ge=0, strict=True)


class SyntheticEvaluationResult(OperationModel):
    """Serializable metrics and provenance, separate from arrays and detector records."""

    scenario_id: str
    generator_config: SyntheticRecordingConfig
    random_seed: int = Field(ge=0, strict=True)
    dataset_reference: DatasetReference
    detector_operation_id: str
    detector_operation_version: str
    detector_implementation_version: str
    detector_parameters: NegativeSpikeParameters
    metadata_snapshot: tuple[MetadataSnapshot, ...]
    execution_ids: tuple[str, ...]
    matching_tolerance_samples: int = Field(ge=0, strict=True)
    event_evaluation: EventEvaluation
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


class SyntheticEvaluationExecutionError(RuntimeError):
    """A detector operation failed during an approved synthetic evaluation."""

    def __init__(self, record: ExecutionRecord) -> None:
        self.record = record
        detail = record.error.message if record.error is not None else "unknown failure"
        super().__init__(f"synthetic detector execution failed: {detail}")


ApprovalProvider = Callable[
    [OperationRunner, PreparedExecution], ApprovedExecution
]


def _best_ordered_pairs(
    ground_truth: Sequence[tuple[int, GroundTruthSpike]],
    detections: Sequence[tuple[int, DetectionEvent]],
    tolerance_samples: int,
) -> tuple[tuple[int, int], ...]:
    """DP maximizes cardinality, then minimizes error, with fixed local tie order.

    Inputs are sorted by sample index before this helper is called. In one
    dimension, crossing pairs can be uncrossed without losing eligibility and
    without increasing the sum of absolute errors, so an optimal ordered
    matching exists. The recurrence considers skipping either endpoint or
    matching an eligible pair. Exact objective ties prefer skipping the latest
    sorted detection, then the latest sorted ground-truth event, then matching
    the final eligible pair. This makes duplicate-time outcomes reproducible.
    Time and memory are O(n*m) for n target events and m detections.
    """
    n = len(ground_truth)
    m = len(detections)
    counts = [[0] * (m + 1) for _ in range(n + 1)]
    errors = [[0] * (m + 1) for _ in range(n + 1)]
    choices = [[0] * (m + 1) for _ in range(n + 1)]
    skip_ground_truth = 1
    skip_detection = 2
    match = 3
    for i in range(1, n + 1):
        choices[i][0] = skip_ground_truth
    for j in range(1, m + 1):
        choices[0][j] = skip_detection

    priorities = {
        skip_detection: 0,
        skip_ground_truth: 1,
        match: 2,
    }
    for i in range(1, n + 1):
        ground_truth_sample = ground_truth[i - 1][1].sample_index
        for j in range(1, m + 1):
            candidates = [
                (counts[i][j - 1], errors[i][j - 1], priorities[skip_detection], skip_detection),
                (counts[i - 1][j], errors[i - 1][j], priorities[skip_ground_truth], skip_ground_truth),
            ]
            detection_sample = detections[j - 1][1].original_sample_index
            error = abs(detection_sample - ground_truth_sample)
            if error <= tolerance_samples:
                candidates.append(
                    (
                        counts[i - 1][j - 1] + 1,
                        errors[i - 1][j - 1] + error,
                        priorities[match],
                        match,
                    )
                )
            best = min(candidates, key=lambda item: (-item[0], item[1], item[2]))
            counts[i][j], errors[i][j], _, choices[i][j] = best

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i or j:
        choice = choices[i][j]
        if choice == match:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif choice == skip_detection:
            j -= 1
        elif choice == skip_ground_truth:
            i -= 1
        else:
            raise RuntimeError("matching dynamic program produced an invalid backtrace")
    pairs.reverse()
    return tuple(pairs)


def match_events_one_to_one(
    ground_truth: Sequence[GroundTruthSpike],
    detections: Sequence[DetectionEvent],
    *,
    tolerance_samples: int,
    target_polarity: str = "negative",
) -> tuple[MatchedEvent, ...]:
    """Match one target polarity within an explicit inclusive sample tolerance."""
    _validate_tolerance(tolerance_samples)
    if target_polarity not in ("negative", "positive"):
        raise ValueError("target_polarity must be 'negative' or 'positive'.")
    target_truth = [
        (index, event)
        for index, event in enumerate(ground_truth)
        if event.polarity == target_polarity
    ]
    indexed_detections = list(enumerate(detections))
    channels = sorted(
        {event.channel_index for _, event in target_truth}
        | {event.channel_index for _, event in indexed_detections}
    )
    results: list[MatchedEvent] = []
    for channel in channels:
        channel_truth = sorted(
            (pair for pair in target_truth if pair[1].channel_index == channel),
            key=lambda pair: (pair[1].sample_index, pair[1].spike_id, pair[0]),
        )
        channel_detections = sorted(
            (pair for pair in indexed_detections if pair[1].channel_index == channel),
            key=lambda pair: (
                pair[1].original_sample_index,
                pair[1].detection_index,
                pair[0],
            ),
        )
        for truth_position, detection_position in _best_ordered_pairs(
            channel_truth, channel_detections, tolerance_samples
        ):
            truth = channel_truth[truth_position][1]
            detection = channel_detections[detection_position][1]
            signed_error = detection.original_sample_index - truth.sample_index
            results.append(
                MatchedEvent(
                    channel_index=channel,
                    ground_truth_event_id=truth.spike_id,
                    ground_truth_sample_index=truth.sample_index,
                    detection_index=detection.detection_index,
                    detected_window_relative_sample_index=detection.window_relative_sample_index,
                    detected_original_sample_index=detection.original_sample_index,
                    signed_timing_error_samples=signed_error,
                    absolute_timing_error_samples=abs(signed_error),
                )
            )
    return tuple(results)


def evaluate_detector_outputs(
    outputs: Sequence[NegativeSpikeOutput],
    ground_truth: Sequence[GroundTruthSpike],
    *,
    matching_tolerance_samples: int,
) -> EventEvaluation:
    """Backward-compatible negative-polarity wrapper over the shared evaluator."""
    if not outputs:
        raise ValueError("at least one detector output is required for evaluation.")
    interval = outputs[0].sample_interval
    detections: list[DetectionEvent] = []
    detection_channels: list[int] = []
    for output in outputs:
        if output.sample_interval != interval:
            raise ValueError("all detector outputs must use the same sample interval.")
        if not (
            len(output.window_relative_sample_indices)
            == len(output.original_sample_indices)
            == len(output.amplitudes)
        ):
            raise ValueError("detector output event arrays have inconsistent lengths.")
        detection_channels.append(output.channel_index)
        detections.extend(
            DetectionEvent(
                detection_index=index,
                channel_index=output.channel_index,
                window_relative_sample_index=relative,
                original_sample_index=original,
            )
            for index, (relative, original) in enumerate(
                zip(output.window_relative_sample_indices, output.original_sample_indices, strict=True)
            )
        )
    rates = {(output.sampling_rate_hz, output.sampling_rate_status) for output in outputs}
    if len(rates) != 1:
        raise ValueError("detector outputs disagree about sampling-rate metadata.")
    rate, rate_status = next(iter(rates))
    generic = evaluate_detection_events(
        detections,
        ground_truth,
        selected_channels=tuple(detection_channels),
        sample_interval=interval,
        sampling_rate_hz=rate,
        sampling_rate_status=rate_status,
        matching_tolerance_samples=matching_tolerance_samples,
        target_polarity="negative",
    )
    channels = tuple(
        ChannelEvaluation(
            channel_index=item.channel_index,
            negative_target_event_count=item.target_event_count,
            positive_excluded_event_count=item.opposite_polarity_excluded_event_count,
            detection_count=item.detection_count,
            metrics=item.metrics,
            matched_events=item.matched_events,
            unmatched_detections=item.unmatched_detections,
            unmatched_ground_truth_events=item.unmatched_ground_truth_events,
            excluded_positive_events=tuple(
                ExcludedPositiveEvent(
                    event_id=event.event_id,
                    channel_index=event.channel_index,
                    sample_index=event.sample_index,
                )
                for event in item.excluded_ground_truth_events
            ),
        )
        for item in generic.per_channel
    )
    return EventEvaluation(
        sample_interval=generic.sample_interval,
        matching_tolerance_samples=generic.matching_tolerance_samples,
        matching_algorithm_id=generic.matching_algorithm_id,
        matching_algorithm_version=generic.matching_algorithm_version,
        total_negative_target_events=generic.total_target_events,
        total_positive_excluded_events=generic.total_opposite_polarity_excluded_events,
        total_detections=generic.total_detections,
        number_of_matches=generic.number_of_matches,
        number_of_unmatched_detections=generic.number_of_unmatched_detections,
        number_of_unmatched_ground_truth_events=generic.number_of_unmatched_ground_truth_events,
        per_channel=channels,
        aggregate_micro_metrics=generic.aggregate_micro_metrics,
        warnings=generic.warnings,
        limitations=generic.limitations,
    )


def evaluate_detection_events(
    detections: Sequence[DetectionEvent],
    ground_truth: Sequence[GroundTruthSpike],
    *,
    selected_channels: Sequence[int],
    sample_interval: SampleInterval,
    sampling_rate_hz: float | None,
    sampling_rate_status: MetadataStatus,
    matching_tolerance_samples: int,
    target_polarity: Literal["negative", "positive"],
) -> PolarityEventEvaluation:
    """Shared polarity-aware matching and micro-metric computation."""
    _validate_tolerance(matching_tolerance_samples)
    if target_polarity not in ("negative", "positive"):
        raise ValueError("target_polarity must be 'negative' or 'positive'.")
    channels_selected = tuple(selected_channels)
    if not channels_selected or len(set(channels_selected)) != len(channels_selected):
        raise ValueError("selected_channels must be non-empty and unique.")
    if any(
        isinstance(channel, bool) or not isinstance(channel, int) or channel < 0
        for channel in channels_selected
    ):
        raise ValueError("selected_channels must contain non-negative integer indices.")
    if len({event.spike_id for event in ground_truth}) != len(ground_truth):
        raise ValueError("ground-truth event IDs must be unique.")
    if sampling_rate_hz is not None and (
        not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0
    ):
        raise ValueError("sampling rate, when supplied, must be finite and positive.")
    if sampling_rate_status is MetadataStatus.CONFIRMED:
        if sampling_rate_hz is None or not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
            raise ValueError("confirmed sampling rate must be available and positive.")
        error_scale = float(sampling_rate_hz)
    else:
        error_scale = None
    channel_set = set(channels_selected)
    for detection in detections:
        if detection.channel_index not in channel_set:
            raise ValueError("detection channel is not included in selected_channels.")
        if detection.original_sample_index != sample_interval.start + detection.window_relative_sample_index:
            raise ValueError("detector sample indices do not map to the original interval.")
        if not sample_interval.start <= detection.original_sample_index < sample_interval.stop:
            raise ValueError("detector event lies outside its declared sample interval.")
    for channel in channels_selected:
        detection_indices = [
            item.detection_index
            for item in detections
            if item.channel_index == channel
        ]
        if len(set(detection_indices)) != len(detection_indices):
            raise ValueError("detection indices must be unique within each channel.")
    in_interval = [
        event
        for event in ground_truth
        if event.channel_index in channel_set
        and sample_interval.start <= event.sample_index < sample_interval.stop
    ]
    if any(event.polarity not in ("negative", "positive") for event in in_interval):
        raise ValueError("ground-truth event polarity must be positive or negative.")
    targets = [event for event in in_interval if event.polarity == target_polarity]
    excluded = [event for event in in_interval if event.polarity != target_polarity]

    channels: list[PolarityChannelEvaluation] = []
    all_matches: list[MatchedEvent] = []
    all_unmatched_detections: list[DetectionEvent] = []
    all_unmatched_truth: list[UnmatchedGroundTruthEvent] = []
    all_excluded: list[ExcludedGroundTruthEvent] = []
    for channel in sorted(channel_set):
        channel_detections = tuple(
            sorted(
                (item for item in detections if item.channel_index == channel),
                key=lambda item: (item.original_sample_index, item.detection_index),
            )
        )
        channel_targets = [event for event in targets if event.channel_index == channel]
        channel_excluded = [event for event in excluded if event.channel_index == channel]
        matches = match_events_one_to_one(
            channel_targets,
            channel_detections,
            tolerance_samples=matching_tolerance_samples,
            target_polarity=target_polarity,
        )
        if error_scale is not None:
            matches = tuple(
                item.model_copy(
                    update={
                        "signed_timing_error_seconds": item.signed_timing_error_samples / error_scale,
                        "absolute_timing_error_seconds": item.absolute_timing_error_samples / error_scale,
                    }
                )
                for item in matches
            )
        matched_detection_indices = {item.detection_index for item in matches}
        matched_truth_ids = {item.ground_truth_event_id for item in matches}
        unmatched_detections = tuple(
            item for item in channel_detections if item.detection_index not in matched_detection_indices
        )
        unmatched_truth = tuple(
            UnmatchedGroundTruthEvent(
                event_id=event.spike_id,
                channel_index=event.channel_index,
                sample_index=event.sample_index,
            )
            for event in channel_targets
            if event.spike_id not in matched_truth_ids
        )
        excluded_records = tuple(
            ExcludedGroundTruthEvent(
                event_id=event.spike_id,
                channel_index=event.channel_index,
                sample_index=event.sample_index,
                polarity=event.polarity,
            )
            for event in channel_excluded
        )
        metrics = _calculate_metrics(
            matches,
            target_count=len(channel_targets),
            detection_count=len(channel_detections),
            error_scale=error_scale,
        )
        channels.append(
            PolarityChannelEvaluation(
                channel_index=channel,
                target_event_count=len(channel_targets),
                opposite_polarity_excluded_event_count=len(channel_excluded),
                detection_count=len(channel_detections),
                metrics=metrics,
                matched_events=matches,
                unmatched_detections=unmatched_detections,
                unmatched_ground_truth_events=unmatched_truth,
                excluded_ground_truth_events=excluded_records,
            )
        )
        all_matches.extend(matches)
        all_unmatched_detections.extend(unmatched_detections)
        all_unmatched_truth.extend(unmatched_truth)
        all_excluded.extend(excluded_records)
    total_detections = len(detections)
    aggregate = _calculate_metrics(
        all_matches,
        target_count=len(targets),
        detection_count=total_detections,
        error_scale=error_scale,
    )
    return PolarityEventEvaluation(
        target_polarity=target_polarity,
        sample_interval=sample_interval,
        matching_tolerance_samples=matching_tolerance_samples,
        matching_algorithm_id=MATCHING_ALGORITHM_ID,
        matching_algorithm_version=MATCHING_ALGORITHM_VERSION,
        total_target_events=len(targets),
        total_opposite_polarity_excluded_events=len(excluded),
        total_detections=total_detections,
        number_of_matches=len(all_matches),
        number_of_unmatched_detections=len(all_unmatched_detections),
        number_of_unmatched_ground_truth_events=len(all_unmatched_truth),
        per_channel=tuple(channels),
        aggregate_micro_metrics=aggregate,
        warnings=(
            "Metrics describe this synthetic ground-truth scenario only.",
            f"{('Positive' if target_polarity == 'negative' else 'Negative')}-going events are excluded from target-polarity false negatives.",
        ),
        limitations=(
            "Synthetic evaluation is not biological validation.",
            "Matching is restricted to selected channels and the half-open sample interval.",
        ),
    )


def evaluate_signed_peak_outputs(
    outputs: Sequence[SignedPeakOutput],
    ground_truth: Sequence[GroundTruthSpike],
    *,
    matching_tolerance_samples: int,
    target_polarity: str,
) -> PolarityEventEvaluation:
    """Evaluate owner-detector outputs in original sample coordinates."""
    if not outputs:
        raise ValueError("at least one signed peak output is required for evaluation.")
    if target_polarity not in ("negative", "positive"):
        raise ValueError("target_polarity must be 'negative' or 'positive'.")
    intervals = {item.sample_interval for item in outputs}
    if len(intervals) != 1:
        raise ValueError("all signed peak outputs must use the same sample interval.")
    interval = outputs[0].sample_interval
    rates = {(item.sampling_rate_hz, item.sampling_rate_status) for item in outputs}
    if len(rates) != 1:
        raise ValueError("signed peak outputs disagree about sampling-rate metadata.")
    for output in outputs:
        if output.polarity not in ("none", target_polarity):
            raise ValueError("all signed peak output polarities must match the target polarity.")
        if not (
            len(output.window_relative_sample_indices)
            == len(output.original_sample_indices)
            == len(output.amplitudes)
        ):
            raise ValueError("signed peak output event arrays have inconsistent lengths.")
    detections = tuple(
        DetectionEvent(
            detection_index=index,
            channel_index=output.channel_index,
            window_relative_sample_index=relative,
            original_sample_index=original,
        )
        for output in outputs
        for index, (relative, original) in enumerate(
            zip(output.window_relative_sample_indices, output.original_sample_indices, strict=True)
        )
    )
    rate, rate_status = next(iter(rates))
    return evaluate_detection_events(
        detections,
        ground_truth,
        selected_channels=tuple(output.channel_index for output in outputs),
        sample_interval=interval,
        sampling_rate_hz=rate,
        sampling_rate_status=rate_status,
        matching_tolerance_samples=matching_tolerance_samples,
        target_polarity=target_polarity,
    )


def _calculate_metrics(
    matches: Sequence[MatchedEvent],
    *,
    target_count: int,
    detection_count: int,
    error_scale: float | None,
) -> EvaluationMetrics:
    true_positives = len(matches)
    false_positives = detection_count - true_positives
    false_negatives = target_count - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else None
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else None
    )
    f1 = (
        2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        if precision is not None and recall is not None
        else None
    )
    absolute_samples = [match.absolute_timing_error_samples for match in matches]
    signed_samples = [match.signed_timing_error_samples for match in matches]
    absolute_seconds = (
        [value / error_scale for value in absolute_samples]
        if error_scale is not None
        else []
    )
    signed_seconds = (
        [value / error_scale for value in signed_samples]
        if error_scale is not None
        else []
    )
    return EvaluationMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall_sensitivity=recall,
        f1_score=f1,
        mean_absolute_timing_error_samples=(
            fsum(absolute_samples) / len(absolute_samples) if absolute_samples else None
        ),
        median_absolute_timing_error_samples=(median(absolute_samples) if absolute_samples else None),
        maximum_absolute_timing_error_samples=(max(absolute_samples) if absolute_samples else None),
        mean_signed_timing_error_samples=(
            fsum(signed_samples) / len(signed_samples) if signed_samples else None
        ),
        median_signed_timing_error_samples=(median(signed_samples) if signed_samples else None),
        mean_absolute_timing_error_seconds=(
            fsum(absolute_seconds) / len(absolute_seconds) if absolute_seconds else None
        ),
        median_absolute_timing_error_seconds=(median(absolute_seconds) if absolute_seconds else None),
        maximum_absolute_timing_error_seconds=(max(absolute_seconds) if absolute_seconds else None),
        mean_signed_timing_error_seconds=(
            fsum(signed_seconds) / len(signed_seconds) if signed_seconds else None
        ),
        median_signed_timing_error_seconds=(median(signed_seconds) if signed_seconds else None),
    )


def run_synthetic_scenario(
    scenario: SyntheticEvaluationScenario,
    registry: OperationRegistry,
    *,
    approval_provider: ApprovalProvider,
) -> SyntheticEvaluationResult:
    """Generate, explicitly approve/run the detector per channel, then evaluate.

    The caller supplies an approval provider; this function never auto-approves.
    Ground truth remains outside the dataset store and detector input until every
    approved detector execution has completed.
    """
    if not isinstance(scenario, SyntheticEvaluationScenario):
        raise TypeError("scenario must be a SyntheticEvaluationScenario.")
    if not isinstance(registry, OperationRegistry):
        raise TypeError("registry must be an OperationRegistry.")
    if not callable(approval_provider):
        raise TypeError("approval_provider must explicitly approve each prepared request.")
    definition = registry.get(OPERATION_ID, version=OPERATION_VERSION)
    if definition is not NEGATIVE_SPIKE_DETECTOR:
        raise ValueError("registry entry is not the production negative spike detector.")

    generated = generate_synthetic_recording(scenario.generator_config)
    recording = generated.recording
    profile = DatasetProfile(
        dataset_id=f"synthetic:{scenario.scenario_id}",
        name=scenario.scenario_id,
        source_path=f"synthetic://{scenario.scenario_id}",
        file_format="synthetic",
        shape=list(recording.signal.shape),
        dtype=str(recording.signal.dtype),
        metadata=recording.metadata,
        confirmed_fields=[
            "channel_axis",
            "time_axis",
            "sampling_rate_hz",
            "units",
            "signal_kind",
        ],
        unknown_fields=[],
    )
    exploration = DatasetExploration(recording.signal, profile)
    dataset_reference = DatasetReference(
        dataset_id=profile.dataset_id,
        source_reference=f"synthetic://{scenario.scenario_id}",
    )
    store = DatasetStore()
    registration: DatasetRegistration = store.register(
        exploration,
        dataset=dataset_reference,
    )
    runner = OperationRunner(registry, data_store=store)
    metadata_snapshot = tuple(
        MetadataSnapshot(
            name=name,
            value=exploration.summary.metadata[name].value,
            status=exploration.summary.metadata[name].status,
        )
        for name in ("signal_kind", "sampling_rate_hz", "units")
    )
    context = OperationContext(
        dataset_reference=dataset_reference,
        metadata=metadata_snapshot,
        reproducibility=ReproducibilityInfo(
            seed=scenario.generator_config.seed,
            rng_algorithm="PCG64",
            configuration_reference=f"synthetic-config://{scenario.scenario_id}",
        ),
    )

    detector_outputs: list[NegativeSpikeOutput] = []
    execution_records: list[ExecutionRecord] = []
    for channel_index in range(scenario.generator_config.channel_count):
        reference = registration.reference(
            (channel_index,),
            SampleInterval(start=0, stop=scenario.generator_config.sample_count),
        )
        prepared = runner.prepare(
            OPERATION_ID,
            OPERATION_VERSION,
            NegativeSpikeInput(signal=reference),
            scenario.detector_parameters,
            context=context,
        )
        approval = approval_provider(runner, prepared)
        if not isinstance(approval, ApprovedExecution):
            raise TypeError("approval_provider must return an ApprovedExecution.")
        record = runner.execute(approval)
        execution_records.append(record)
        if record.status is not ExecutionStatus.SUCCEEDED:
            raise SyntheticEvaluationExecutionError(record)
        if not isinstance(record.output, NegativeSpikeOutput):
            raise TypeError("detector execution returned an unexpected output model.")
        detector_outputs.append(record.output)

    # Ground truth enters the evaluator only after all detector calls have ended.
    event_evaluation = evaluate_detector_outputs(
        detector_outputs,
        generated.ground_truth,
        matching_tolerance_samples=scenario.matching_tolerance_samples,
    )
    warnings = tuple(
        dict.fromkeys(
            warning
            for record in execution_records
            for warning in record.warnings
        )
    ) + event_evaluation.warnings
    limitations = tuple(
        dict.fromkeys(
            limitation
            for limitation in (definition.limitations + event_evaluation.limitations)
        )
    )
    return SyntheticEvaluationResult(
        scenario_id=scenario.scenario_id,
        generator_config=scenario.generator_config,
        random_seed=scenario.generator_config.seed,
        dataset_reference=dataset_reference,
        detector_operation_id=definition.operation_id,
        detector_operation_version=definition.version,
        detector_implementation_version=definition.implementation_version,
        detector_parameters=scenario.detector_parameters,
        metadata_snapshot=metadata_snapshot,
        execution_ids=tuple(record.execution_id for record in execution_records),
        matching_tolerance_samples=scenario.matching_tolerance_samples,
        event_evaluation=event_evaluation,
        warnings=warnings,
        limitations=limitations,
    )


def run_synthetic_scenarios(
    scenarios: Sequence[SyntheticEvaluationScenario],
    registry: OperationRegistry,
    *,
    approval_provider: ApprovalProvider,
) -> tuple[SyntheticEvaluationResult, ...]:
    """Run a caller-specified finite set of scenarios without writing/overwriting data."""
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario IDs must be unique within one experiment batch.")
    return tuple(
        run_synthetic_scenario(
            scenario,
            registry,
            approval_provider=approval_provider,
        )
        for scenario in scenarios
    )


def run_synthetic_detector_comparison(
    scenario: SyntheticDetectorComparisonScenario,
    registry: OperationRegistry,
    *,
    approval_provider: ApprovalProvider,
) -> SyntheticDetectorComparisonResult:
    """Run both registered detector operations on one preserved synthetic recording.

    Each channel/run goes through prepare, explicit caller approval, then execute.
    Optional baseline offsets are a separate, recorded analysis-input transform;
    the generated recording and its ground truth are left unchanged.
    """
    if not isinstance(scenario, SyntheticDetectorComparisonScenario):
        raise TypeError("scenario must be a SyntheticDetectorComparisonScenario.")
    if not isinstance(registry, OperationRegistry):
        raise TypeError("registry must be an OperationRegistry.")
    if not callable(approval_provider):
        raise TypeError("approval_provider must explicitly approve each prepared request.")
    negative_definition = registry.get(OPERATION_ID, version=OPERATION_VERSION)
    signed_definition = registry.get(
        SIGNED_PEAK_OPERATION_ID, version=SIGNED_PEAK_OPERATION_VERSION
    )
    if negative_definition is not NEGATIVE_SPIKE_DETECTOR:
        raise ValueError("registry entry is not the production negative spike detector.")
    if signed_definition is not SIGNED_PEAK_DETECTOR:
        raise ValueError("registry entry is not the registered signed peak detector.")

    generated = generate_synthetic_recording(scenario.generator_config)
    recording = generated.recording
    if scenario.baseline_offset_by_channel is None:
        analysis_signal = recording.signal
    else:
        offsets = np.asarray(scenario.baseline_offset_by_channel, dtype=float)
        analysis_signal = np.asarray(recording.signal + offsets[:, None], dtype=float)
        analysis_signal.setflags(write=False)

    metadata = recording.metadata.model_copy(
        update={"filtering_description": scenario.filtering_description}
    )
    profile = DatasetProfile(
        dataset_id=f"synthetic:{scenario.scenario_id}:analysis-input",
        name=scenario.scenario_id,
        source_path=f"synthetic://{scenario.scenario_id}/analysis-input",
        file_format="synthetic",
        shape=list(analysis_signal.shape),
        dtype=str(analysis_signal.dtype),
        metadata=metadata,
        confirmed_fields=[
            "channel_axis",
            "time_axis",
            "sampling_rate_hz",
            "units",
            "signal_kind",
        ]
        + (
            ["filtering_description"]
            if scenario.filtering_status is MetadataStatus.CONFIRMED
            else []
        ),
        inferred_fields=(
            ["filtering_description"]
            if scenario.filtering_status is MetadataStatus.INFERRED
            else []
        ),
        unknown_fields=(
            ["filtering_description"]
            if scenario.filtering_status is MetadataStatus.UNKNOWN
            else []
        ),
    )
    exploration = DatasetExploration(analysis_signal, profile)
    dataset_reference = DatasetReference(
        dataset_id=profile.dataset_id,
        source_reference=f"synthetic://{scenario.scenario_id}/analysis-input",
    )
    store = DatasetStore()
    registration = store.register(exploration, dataset=dataset_reference)
    runner = OperationRunner(registry, data_store=store)
    metadata_snapshot = tuple(
        MetadataSnapshot(
            name=name,
            value=exploration.summary.metadata[name].value,
            status=exploration.summary.metadata[name].status,
        )
        for name in ("signal_kind", "sampling_rate_hz", "units", "filtering_description")
    )
    interval = SampleInterval(start=0, stop=scenario.generator_config.sample_count)
    all_records: list[ExecutionRecord] = []
    per_detector_runs: list[DetectorRunEvaluation] = []

    def run_approved(
        operation_id: str,
        operation_version: str,
        input_model: OperationModel,
        parameters: OperationModel,
        channel_index: int,
    ) -> ExecutionRecord:
        reference = registration.reference((channel_index,), interval)
        channel_context = OperationContext(
            dataset_reference=dataset_reference,
            metadata=metadata_snapshot,
            selected_channels=(channel_index,),
            sample_interval=interval,
            reproducibility=ReproducibilityInfo(
                seed=scenario.generator_config.seed,
                rng_algorithm="PCG64",
                configuration_reference=f"synthetic-config://{scenario.scenario_id}",
            ),
        )
        # Inject only the typed data reference; neither operation receives GT.
        input_payload = input_model.model_copy(update={"signal": reference})
        prepared = runner.prepare(
            operation_id,
            operation_version,
            input_payload,
            parameters,
            context=channel_context,
        )
        approval = approval_provider(runner, prepared)
        if not isinstance(approval, ApprovedExecution):
            raise TypeError("approval_provider must return an ApprovedExecution.")
        record = runner.execute(approval)
        all_records.append(record)
        if record.status is not ExecutionStatus.SUCCEEDED:
            raise SyntheticEvaluationExecutionError(record)
        return record

    negative_outputs: list[NegativeSpikeOutput] = []
    negative_records: list[ExecutionRecord] = []
    for channel_index in range(scenario.generator_config.channel_count):
        record = run_approved(
            OPERATION_ID,
            OPERATION_VERSION,
            NegativeSpikeInput(signal=registration.reference((channel_index,), interval)),
            scenario.negative_detector_parameters,
            channel_index,
        )
        if not isinstance(record.output, NegativeSpikeOutput):
            raise TypeError("negative detector returned an unexpected output model.")
        negative_outputs.append(record.output)
        negative_records.append(record)
    negative_evaluation = evaluate_detector_outputs(
        negative_outputs,
        generated.ground_truth,
        matching_tolerance_samples=scenario.matching_tolerance_samples,
    )
    per_detector_runs.append(
        DetectorRunEvaluation(
            run_id="negative-mad",
            operation_id=OPERATION_ID,
            operation_version=negative_definition.version,
            implementation_version=negative_definition.implementation_version,
            parameter_snapshot=scenario.negative_detector_parameters,
            polarity_evaluation=_negative_evaluation_as_polarity(negative_evaluation),
            channel_summaries=tuple(
                ChannelDetectionSummary(
                    channel_index=item.channel_index,
                    threshold_mode="median_centered_mad_negative_multiplier",
                    polarity="negative",
                    noise_estimate=item.noise_sigma,
                    noise_source="estimated",
                    threshold=item.threshold,
                    detection_count=len(item.original_sample_indices),
                )
                for item in negative_outputs
            ),
            execution_ids=tuple(item.execution_id for item in negative_records),
            algorithm_provenance=None,
            warnings=tuple(dict.fromkeys(
                warning for item in negative_records for warning in item.warnings
            )) + negative_evaluation.warnings,
            limitations=negative_definition.limitations + negative_evaluation.limitations,
        )
    )

    for run in scenario.signed_peak_runs:
        run_outputs: list[SignedPeakOutput] = []
        run_records: list[ExecutionRecord] = []
        for channel_index in range(scenario.generator_config.channel_count):
            record = run_approved(
                SIGNED_PEAK_OPERATION_ID,
                SIGNED_PEAK_OPERATION_VERSION,
                SignedPeakInput(signal=registration.reference((channel_index,), interval)),
                run.parameters,
                channel_index,
            )
            if not isinstance(record.output, SignedPeakOutput):
                raise TypeError("signed peak detector returned an unexpected output model.")
            run_outputs.append(record.output)
            run_records.append(record)
        polarity_evaluation = evaluate_signed_peak_outputs(
            run_outputs,
            generated.ground_truth,
            matching_tolerance_samples=scenario.matching_tolerance_samples,
            target_polarity=run.target_polarity,
        )
        per_detector_runs.append(
            DetectorRunEvaluation(
                run_id=run.run_id,
                operation_id=SIGNED_PEAK_OPERATION_ID,
                operation_version=signed_definition.version,
                implementation_version=signed_definition.implementation_version,
                parameter_snapshot=run.parameters,
                polarity_evaluation=polarity_evaluation,
                channel_summaries=tuple(
                    ChannelDetectionSummary(
                        channel_index=item.channel_index,
                        threshold_mode=item.threshold_mode,
                        polarity=item.polarity,
                        noise_estimate=item.noise_estimate,
                        noise_source=item.noise_source,
                        threshold=item.threshold,
                        detection_count=len(item.original_sample_indices),
                    )
                    for item in run_outputs
                ),
                execution_ids=tuple(item.execution_id for item in run_records),
                algorithm_provenance=run_outputs[0].algorithm_provenance,
                warnings=tuple(dict.fromkeys(
                    warning for item in run_records for warning in item.warnings
                )) + polarity_evaluation.warnings,
                limitations=signed_definition.limitations + polarity_evaluation.limitations,
            )
        )

    warnings = list(
        dict.fromkeys(warning for record in all_records for warning in record.warnings)
    )
    if scenario.baseline_offset_by_channel is not None:
        warnings.append(
            "A separately configured baseline offset was applied to the analysis input; the generated recording and ground truth remain unchanged."
        )
    warnings.append("Synthetic evaluation does not establish biological validity.")
    return SyntheticDetectorComparisonResult(
        scenario_id=scenario.scenario_id,
        generator_config=scenario.generator_config,
        random_seed=scenario.generator_config.seed,
        baseline_offset_by_channel=scenario.baseline_offset_by_channel,
        dataset_reference=dataset_reference,
        metadata_snapshot=metadata_snapshot,
        matching_tolerance_samples=scenario.matching_tolerance_samples,
        negative_detector=per_detector_runs[0],
        signed_peak_runs=tuple(per_detector_runs[1:]),
        warnings=tuple(dict.fromkeys(warnings)),
        limitations=tuple(
            dict.fromkeys(
                limitation
                for run_result in per_detector_runs
                for limitation in run_result.limitations
            )
        ),
    )


def _negative_evaluation_as_polarity(
    evaluation: EventEvaluation,
) -> PolarityEventEvaluation:
    """Adapt the stable legacy negative-evaluation shape to the shared contract."""
    return PolarityEventEvaluation(
        target_polarity="negative",
        sample_interval=evaluation.sample_interval,
        matching_tolerance_samples=evaluation.matching_tolerance_samples,
        matching_algorithm_id=evaluation.matching_algorithm_id,
        matching_algorithm_version=evaluation.matching_algorithm_version,
        total_target_events=evaluation.total_negative_target_events,
        total_opposite_polarity_excluded_events=evaluation.total_positive_excluded_events,
        total_detections=evaluation.total_detections,
        number_of_matches=evaluation.number_of_matches,
        number_of_unmatched_detections=evaluation.number_of_unmatched_detections,
        number_of_unmatched_ground_truth_events=evaluation.number_of_unmatched_ground_truth_events,
        per_channel=tuple(
            PolarityChannelEvaluation(
                channel_index=item.channel_index,
                target_event_count=item.negative_target_event_count,
                opposite_polarity_excluded_event_count=item.positive_excluded_event_count,
                detection_count=item.detection_count,
                metrics=item.metrics,
                matched_events=item.matched_events,
                unmatched_detections=item.unmatched_detections,
                unmatched_ground_truth_events=item.unmatched_ground_truth_events,
                excluded_ground_truth_events=tuple(
                    ExcludedGroundTruthEvent(
                        event_id=event.event_id,
                        channel_index=event.channel_index,
                        sample_index=event.sample_index,
                        polarity="positive",
                    )
                    for event in item.excluded_positive_events
                ),
            )
            for item in evaluation.per_channel
        ),
        aggregate_micro_metrics=evaluation.aggregate_micro_metrics,
        warnings=evaluation.warnings,
        limitations=evaluation.limitations,
    )


def _validate_tolerance(tolerance_samples: int) -> None:
    if isinstance(tolerance_samples, bool) or not isinstance(tolerance_samples, int):
        raise TypeError("matching tolerance must be an integer number of samples.")
    if tolerance_samples < 0:
        raise ValueError("matching tolerance must be non-negative.")
