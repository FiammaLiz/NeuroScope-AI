
import numpy as np
import pytest
from pydantic import ValidationError

from neuroscope.analysis.contracts import (
    SampleInterval,
)
from neuroscope.analysis.evaluation import (
    MATCHING_ALGORITHM_ID,
    DetectionEvent,
    SignedPeakRunConfiguration,
    SyntheticDetectorComparisonScenario,
    SyntheticEvaluationScenario,
    evaluate_detection_events,
    evaluate_detector_outputs,
    match_events_one_to_one,
    run_synthetic_detector_comparison,
    run_synthetic_scenario,
    run_synthetic_scenarios,
)
from neuroscope.analysis.operations import (
    ApprovedExecution,
    OperationRunner,
    PreparedExecution,
)
from neuroscope.analysis.registry import OperationRegistry
from neuroscope.analysis.signed_peak_operation import (
    SignedPeakParameters,
    register_signed_peak_detector,
)
from neuroscope.analysis.spike_operation import (
    OPERATION_ID,
    NegativeSpikeOutput,
    NegativeSpikeParameters,
    register_negative_spike_detector,
)
from neuroscope.data.synthetic import (
    GroundTruthSpike,
    SpikeTemplate,
    SyntheticRecordingConfig,
    SyntheticSpike,
    generate_synthetic_recording,
)
from neuroscope.exploration import MetadataStatus


def _truth(event_id, channel, sample, polarity="negative"):
    return GroundTruthSpike(
        spike_id=event_id,
        channel_index=channel,
        sample_index=sample,
        timestamp_s=float(sample) / 1000,
        amplitude=-1.0 if polarity == "negative" else 1.0,
        polarity=polarity,
        template_id="template",
    )


def _detection(index, channel, sample, *, start=0):
    return DetectionEvent(
        detection_index=index,
        channel_index=channel,
        window_relative_sample_index=sample - start,
        original_sample_index=sample,
    )


def _output(
    channel,
    relative_samples=(),
    *,
    interval=None,
    rate=None,
    rate_status=MetadataStatus.UNKNOWN,
):
    if interval is None:
        interval = SampleInterval(start=0, stop=100)
    original = tuple(interval.start + value for value in relative_samples)
    times = (
        tuple(value / rate for value in relative_samples)
        if rate_status is MetadataStatus.CONFIRMED and rate is not None
        else None
    )
    return NegativeSpikeOutput(
        window_relative_sample_indices=tuple(relative_samples),
        original_sample_indices=original,
        times_s_relative_to_window=times,
        amplitudes=tuple(-1.0 for _ in relative_samples),
        noise_sigma=0.0,
        threshold=0.0,
        threshold_sd=-4.0,
        channel_index=channel,
        sample_interval=interval,
        sampling_rate_hz=rate,
        sampling_rate_status=rate_status,
        amplitude_units=None,
        amplitude_units_status=MetadataStatus.UNKNOWN,
    )


def _config(*, noise=(0.0, 0.0), seed=17):
    return SyntheticRecordingConfig(
        channel_count=2,
        duration_s=0.2,
        sampling_rate_hz=1000.0,
        voltage_units="mV",
        seed=seed,
        noise_std_by_channel=noise,
        templates=(SpikeTemplate("unit", (0.0, 1.0, 0.0), 1),),
        spikes=(
            # One negative target on each channel and a positive event to exclude.
            SyntheticSpike("neg-0", 0, 0.05, -5.0, "unit"),
            SyntheticSpike("neg-1", 1, 0.10, -5.0, "unit"),
            SyntheticSpike("pos-0", 0, 0.15, 5.0, "unit"),
        ),
    )


def _registry():
    registry = OperationRegistry()
    register_negative_spike_detector(registry)
    return registry


def _comparison_registry():
    registry = _registry()
    register_signed_peak_detector(registry)
    return registry


def _scenario(scenario_id, *, noise=(0.0, 0.0), seed=17, threshold=-4.0, tolerance=0):
    return SyntheticEvaluationScenario(
        scenario_id=scenario_id,
        generator_config=_config(noise=noise, seed=seed),
        detector_parameters=NegativeSpikeParameters(threshold_sd=threshold),
        matching_tolerance_samples=tolerance,
    )


def _approve(runner: OperationRunner, prepared: PreparedExecution) -> ApprovedExecution:
    # Explicit test approval; the evaluator itself never auto-approves.
    return runner.approve(prepared, approved_by="synthetic-evaluation-test")


def test_matching_exact_tolerance_and_outside_tolerance():
    truth = (_truth("a", 0, 10), _truth("b", 0, 20))
    detections = (_detection(0, 0, 10), _detection(1, 0, 22), _detection(2, 0, 30))

    matches = match_events_one_to_one(truth, detections, tolerance_samples=2)

    assert [(item.ground_truth_event_id, item.detected_original_sample_index) for item in matches] == [
        ("a", 10),
        ("b", 22),
    ]
    assert [item.absolute_timing_error_samples for item in matches] == [0, 2]


def test_multiple_detections_for_one_ground_truth_leave_extra_detection_unmatched():
    truth = (_truth("single", 0, 10),)
    detections = (_detection(0, 0, 9), _detection(1, 0, 10))

    matches = match_events_one_to_one(truth, detections, tolerance_samples=1)

    assert len(matches) == 1
    assert matches[0].detected_original_sample_index == 10


def test_multiple_ground_truth_events_for_one_detection_leave_one_false_negative():
    truth = (_truth("first", 0, 10), _truth("second", 0, 11))
    detections = (_detection(0, 0, 10),)

    matches = match_events_one_to_one(truth, detections, tolerance_samples=1)

    assert len(matches) == 1
    assert matches[0].ground_truth_event_id == "first"


def test_one_to_one_matching_maximizes_cardinality_where_greedy_can_fail():
    truth = (_truth("early", 0, 10), _truth("late", 0, 11))
    detections = (_detection(0, 0, 6), _detection(1, 0, 10))

    matches = match_events_one_to_one(truth, detections, tolerance_samples=4)

    assert [(item.ground_truth_event_id, item.detected_original_sample_index) for item in matches] == [
        ("early", 6),
        ("late", 10),
    ]
    assert len(matches) == 2


def test_matching_minimizes_total_error_after_maximizing_match_count():
    truth = (_truth("g0", 0, 10), _truth("g1", 0, 14))
    detections = (
        _detection(0, 0, 11),
        _detection(1, 0, 12),
        _detection(2, 0, 17),
    )

    matches = match_events_one_to_one(truth, detections, tolerance_samples=3)

    assert [(item.ground_truth_event_id, item.detected_original_sample_index) for item in matches] == [
        ("g0", 11),
        ("g1", 12),
    ]
    assert sum(item.absolute_timing_error_samples for item in matches) == 3


def test_equal_objective_matching_uses_documented_deterministic_tie_break():
    truth = (_truth("event", 0, 10),)
    detections = (_detection(0, 0, 9), _detection(1, 0, 11))

    first = match_events_one_to_one(truth, detections, tolerance_samples=1)
    second = match_events_one_to_one(truth, detections, tolerance_samples=1)

    assert first == second
    # Tie policy skips the latest sorted detection first, retaining sample 9.
    assert first[0].detected_original_sample_index == 9


def test_duplicates_at_same_sample_remain_one_to_one_and_stable():
    truth = (_truth("a", 0, 10), _truth("b", 0, 10))
    detections = (_detection(0, 0, 10),)

    matches = match_events_one_to_one(truth, detections, tolerance_samples=0)

    assert len(matches) == 1
    assert matches[0].ground_truth_event_id == "a"


def test_matching_never_pairs_across_channels_and_handles_empty_inputs():
    truth = (_truth("ch0", 0, 10),)
    detections = (_detection(0, 1, 10),)

    assert match_events_one_to_one(truth, detections, tolerance_samples=0) == ()
    assert match_events_one_to_one((), (), tolerance_samples=0) == ()


def test_metrics_use_micro_aggregation_and_exclude_positive_events_from_fn():
    outputs = (
        _output(0, (10, 22, 40), rate=1000.0, rate_status=MetadataStatus.CONFIRMED),
        _output(1, (6,), rate=1000.0, rate_status=MetadataStatus.CONFIRMED),
    )
    truth = (
        _truth("g0", 0, 10),
        _truth("g1", 0, 20),
        _truth("positive", 0, 25, polarity="positive"),
        _truth("g2", 1, 5),
    )

    result = evaluate_detector_outputs(outputs, truth, matching_tolerance_samples=1)

    assert result.total_negative_target_events == 3
    assert result.total_positive_excluded_events == 1
    assert result.total_detections == 4
    assert result.number_of_matches == 2
    assert result.number_of_unmatched_detections == 2
    assert result.number_of_unmatched_ground_truth_events == 1
    channel0, channel1 = result.per_channel
    assert channel0.metrics.true_positives == 1
    assert channel0.metrics.false_positives == 2
    assert channel0.metrics.false_negatives == 1
    assert channel0.positive_excluded_event_count == 1
    assert channel1.metrics.true_positives == 1
    assert channel1.metrics.false_positives == 0
    assert channel1.metrics.false_negatives == 0
    aggregate = result.aggregate_micro_metrics
    assert (aggregate.true_positives, aggregate.false_positives, aggregate.false_negatives) == (2, 2, 1)
    assert aggregate.precision == pytest.approx(0.5)
    assert aggregate.recall_sensitivity == pytest.approx(2 / 3)
    assert aggregate.f1_score == pytest.approx(4 / 7)
    assert aggregate.mean_absolute_timing_error_samples == pytest.approx(0.5)
    assert aggregate.median_absolute_timing_error_samples == pytest.approx(0.5)
    assert aggregate.maximum_absolute_timing_error_samples == 1
    assert aggregate.mean_signed_timing_error_samples == pytest.approx(0.5)
    assert aggregate.mean_absolute_timing_error_seconds == pytest.approx(0.0005)
    assert all(item.event_id != "positive" for item in result.per_channel[0].unmatched_ground_truth_events)
    assert result.matching_algorithm_id == MATCHING_ALGORITHM_ID


def test_zero_denominators_are_none_and_timing_stats_require_matches():
    interval = SampleInterval(start=0, stop=10)
    no_events = evaluate_detector_outputs(
        (_output(0, interval=interval),), (), matching_tolerance_samples=0
    ).aggregate_micro_metrics
    only_detection = evaluate_detector_outputs(
        (_output(0, (3,), interval=interval),), (), matching_tolerance_samples=0
    ).aggregate_micro_metrics
    only_truth = evaluate_detector_outputs(
        (_output(0, interval=interval),),
        (_truth("miss", 0, 3),),
        matching_tolerance_samples=0,
    ).aggregate_micro_metrics
    assert no_events.precision is None
    assert no_events.recall_sensitivity is None
    assert no_events.f1_score is None
    assert no_events.mean_absolute_timing_error_samples is None
    assert only_detection.precision == 0.0
    assert only_detection.recall_sensitivity is None
    assert only_detection.f1_score is None
    assert only_truth.true_positives == 0
    assert only_truth.false_negatives == 1
    assert only_truth.precision is None
    assert only_truth.recall_sensitivity == 0.0
    assert only_truth.f1_score is None


def test_unmatched_detection_is_false_positive_and_unmatched_truth_is_false_negative():
    result = evaluate_detector_outputs(
        (_output(0, (7,)),),
        (_truth("missed", 0, 3),),
        matching_tolerance_samples=0,
    )

    metrics = result.aggregate_micro_metrics
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (0, 1, 1)
    assert result.per_channel[0].unmatched_detections[0].original_sample_index == 7
    assert result.per_channel[0].unmatched_ground_truth_events[0].event_id == "missed"
    assert metrics.precision == 0.0
    assert metrics.recall_sensitivity == 0.0
    assert metrics.f1_score == 0.0


def test_subinterval_uses_absolute_indices_and_half_open_ground_truth_filter():
    interval = SampleInterval(start=10, stop=20)
    output = _output(0, (0, 5), interval=interval, rate=1000.0, rate_status=MetadataStatus.CONFIRMED)
    truth = (
        _truth("before", 0, 9),
        _truth("at-start", 0, 10),
        _truth("inside", 0, 15),
        _truth("at-stop", 0, 20),
        _truth("positive", 0, 12, polarity="positive"),
    )

    result = evaluate_detector_outputs((output,), truth, matching_tolerance_samples=0)

    assert result.sample_interval == interval
    assert result.total_negative_target_events == 2
    assert result.total_positive_excluded_events == 1
    assert result.number_of_matches == 2
    assert [item.detected_window_relative_sample_index for item in result.per_channel[0].matched_events] == [0, 5]
    assert [item.detected_original_sample_index for item in result.per_channel[0].matched_events] == [10, 15]
    assert result.aggregate_micro_metrics.mean_absolute_timing_error_seconds == 0.0


def test_unknown_sampling_rate_keeps_sample_errors_but_no_seconds():
    output = _output(0, (11,), rate=None, rate_status=MetadataStatus.UNKNOWN)
    result = evaluate_detector_outputs(
        (output,), (_truth("event", 0, 10),), matching_tolerance_samples=1
    )

    match = result.per_channel[0].matched_events[0]
    assert match.signed_timing_error_samples == 1
    assert match.signed_timing_error_seconds is None
    assert result.aggregate_micro_metrics.mean_absolute_timing_error_samples == 1.0
    assert result.aggregate_micro_metrics.mean_absolute_timing_error_seconds is None


def test_inconsistent_detector_indices_are_rejected():
    output = _output(0, (4,))
    output = output.model_copy(update={"original_sample_indices": (9,)})

    with pytest.raises(ValueError, match="do not map"):
        evaluate_detector_outputs((output,), (), matching_tolerance_samples=0)


def test_generator_reproduces_recording_and_ground_truth_from_explicit_seed():
    config = _config(noise=(0.01, 0.02), seed=42)
    first = generate_synthetic_recording(config)
    second = generate_synthetic_recording(config)

    np.testing.assert_array_equal(first.recording.signal, second.recording.signal)
    assert first.ground_truth == second.ground_truth


def test_end_to_end_uses_approval_runner_and_keeps_ground_truth_out_of_detector(monkeypatch):
    import neuroscope.analysis.evaluation as evaluation_module

    scenario = _scenario("end-to-end", noise=(0.0, 0.0), seed=19, threshold=-4.0, tolerance=0)
    registry = _registry()
    approvals = []
    captured = []
    original_generator = evaluation_module.generate_synthetic_recording

    def capture_generator(config):
        generated = original_generator(config)
        captured.append(
            (
                generated.recording.signal,
                generated.recording.signal.copy(),
                generated.ground_truth,
            )
        )
        return generated

    monkeypatch.setattr(evaluation_module, "generate_synthetic_recording", capture_generator)

    def approve(runner: OperationRunner, prepared: PreparedExecution):
        assert prepared.operation_id == OPERATION_ID
        assert set(type(prepared.input_data).model_fields) == {"signal"}
        assert not hasattr(prepared.input_data, "ground_truth")
        approvals.append(prepared.input_data.signal)
        return runner.approve(prepared, approved_by="explicit-test-researcher")

    result = run_synthetic_scenario(scenario, registry, approval_provider=approve)

    assert len(approvals) == scenario.generator_config.channel_count
    assert result.detector_operation_id == OPERATION_ID
    assert result.detector_operation_version == "1.0.0"
    assert result.detector_parameters.threshold_sd == -4.0
    assert result.random_seed == 19
    assert result.generator_config == scenario.generator_config
    assert result.event_evaluation.total_negative_target_events == 2
    assert result.event_evaluation.total_positive_excluded_events == 1
    assert result.event_evaluation.aggregate_micro_metrics.true_positives == 2
    assert result.event_evaluation.aggregate_micro_metrics.false_negatives == 0
    assert len(result.execution_ids) == 2
    assert all(snapshot.status is MetadataStatus.CONFIRMED for snapshot in result.metadata_snapshot)
    np.testing.assert_array_equal(captured[0][0], captured[0][1])
    assert len(captured[0][2]) == 3
    assert all(np.asarray(approval.channels).shape == (1,) for approval in approvals)
    assert "biological validation" in " ".join(result.limitations).lower()


def test_scenario_batch_preserves_noise_threshold_seed_and_repeated_results(monkeypatch):
    scenarios = (
        _scenario("quiet-seed-a", noise=(0.0, 0.0), seed=1, threshold=-4.0, tolerance=0),
        _scenario("noisy-seed-b", noise=(0.02, 0.04), seed=2, threshold=-6.0, tolerance=1),
    )
    registry = _registry()
    approvals = []

    def approve(runner, prepared):
        approvals.append(prepared.parameters.threshold_sd)
        return runner.approve(prepared, approved_by="scenario-test")

    first = run_synthetic_scenarios(scenarios, registry, approval_provider=approve)
    first_snapshot = first[0].model_dump_json()
    second = run_synthetic_scenario(scenarios[0], registry, approval_provider=approve)

    assert [item.random_seed for item in first] == [1, 2]
    assert [item.generator_config.noise_std_by_channel for item in first] == [
        (0.0, 0.0),
        (0.02, 0.04),
    ]
    assert [item.detector_parameters.threshold_sd for item in first] == [-4.0, -6.0]
    assert [item.matching_tolerance_samples for item in first] == [0, 1]
    assert first[0].event_evaluation == second.event_evaluation
    assert first[0].model_dump_json() == first_snapshot
    assert first[0].dataset_reference.dataset_id == second.dataset_reference.dataset_id
    assert first[0].execution_ids != second.execution_ids
    assert len(approvals) == 6


def test_scenarios_require_unique_ids_and_no_tolerance_default():
    config = _config()
    with pytest.raises(ValidationError):
        SyntheticEvaluationScenario(
            scenario_id="missing-tolerance",
            generator_config=config,
            detector_parameters=NegativeSpikeParameters(threshold_sd=-4.0),
        )
    scenario = _scenario("same")
    with pytest.raises(ValueError, match="unique"):
        run_synthetic_scenarios(
            (scenario, scenario),
            _registry(),
            approval_provider=_approve,
        )


def test_shared_matcher_evaluates_positive_and_negative_targets_separately():
    interval = SampleInterval(start=10, stop=30)
    detections = (
        _detection(0, 0, 12, start=10),
        _detection(1, 0, 22, start=10),
    )
    truth = (
        _truth("neg", 0, 12, polarity="negative"),
        _truth("pos", 0, 22, polarity="positive"),
    )

    negative = evaluate_detection_events(
        detections,
        truth,
        selected_channels=(0,),
        sample_interval=interval,
        sampling_rate_hz=None,
        sampling_rate_status=MetadataStatus.UNKNOWN,
        matching_tolerance_samples=0,
        target_polarity="negative",
    )
    positive = evaluate_detection_events(
        detections,
        truth,
        selected_channels=(0,),
        sample_interval=interval,
        sampling_rate_hz=None,
        sampling_rate_status=MetadataStatus.UNKNOWN,
        matching_tolerance_samples=0,
        target_polarity="positive",
    )

    assert negative.target_polarity == "negative"
    assert positive.target_polarity == "positive"
    assert negative.aggregate_micro_metrics.true_positives == 1
    assert positive.aggregate_micro_metrics.true_positives == 1
    assert negative.total_opposite_polarity_excluded_events == 1
    assert positive.total_opposite_polarity_excluded_events == 1
    assert negative.aggregate_micro_metrics.false_positives == 1
    assert positive.aggregate_micro_metrics.false_positives == 1
    assert negative.aggregate_micro_metrics.false_negatives == 0
    assert positive.aggregate_micro_metrics.false_negatives == 0


def _comparison_config(*, noise=0.1, seed=5, offset=None, filtering=None):
    from neuroscope.data.synthetic import SyntheticSpike

    return SyntheticDetectorComparisonScenario(
        scenario_id=f"comparison-{seed}-{noise}-{offset}",
        generator_config=SyntheticRecordingConfig(
            channel_count=1,
            duration_s=0.12,
            sampling_rate_hz=1000.0,
            voltage_units="mV",
            seed=seed,
            noise_std_by_channel=(noise,),
            templates=(SpikeTemplate("unit", (0.2, 1.0, 0.2), 1),),
            spikes=(
                SyntheticSpike("negative", 0, 0.04, -1.0, "unit"),
                SyntheticSpike("positive", 0, 0.08, 1.0, "unit"),
            ),
        ),
        negative_detector_parameters=NegativeSpikeParameters(threshold_sd=-4.0),
        signed_peak_runs=(
            SignedPeakRunConfiguration(
                run_id="owner-negative",
                target_polarity="negative",
                parameters=SignedPeakParameters(
                    threshold_mode="absolute", threshold=-0.2, noise_override=1.0
                ),
            ),
            SignedPeakRunConfiguration(
                run_id="owner-positive",
                target_polarity="positive",
                parameters=SignedPeakParameters(
                    threshold_mode="absolute", threshold=0.2, noise_override=1.0
                ),
            ),
        ),
        matching_tolerance_samples=0,
        baseline_offset_by_channel=offset,
        filtering_description=filtering,
        filtering_status=(
            MetadataStatus.CONFIRMED if filtering is not None else MetadataStatus.UNKNOWN
        ),
    )


def test_comparison_runs_both_operations_with_explicit_approval_and_polarity_metrics(monkeypatch):
    import neuroscope.analysis.evaluation as evaluation_module

    scenario = _comparison_config()
    registry = _comparison_registry()
    approvals = []
    captured = []
    original_generator = evaluation_module.generate_synthetic_recording

    def capture(config):
        generated = original_generator(config)
        captured.append((generated.recording.signal, generated.recording.signal.copy()))
        return generated

    monkeypatch.setattr(evaluation_module, "generate_synthetic_recording", capture)

    def approve(runner, prepared):
        approvals.append(prepared)
        assert prepared.context.dataset_reference.dataset_id == scenario.scenario_id.replace(
            "comparison-", "synthetic:comparison-"
        ) + ":analysis-input"
        return runner.approve(prepared, approved_by="comparison-test-researcher")

    result = run_synthetic_detector_comparison(
        scenario, registry, approval_provider=approve
    )

    assert len(approvals) == 3
    assert result.negative_detector.operation_id == OPERATION_ID
    assert result.negative_detector.polarity_evaluation.target_polarity == "negative"
    assert [item.run_id for item in result.signed_peak_runs] == [
        "owner-negative",
        "owner-positive",
    ]
    assert [item.polarity_evaluation.target_polarity for item in result.signed_peak_runs] == [
        "negative",
        "positive",
    ]
    assert [
        item.polarity_evaluation.aggregate_micro_metrics.true_positives
        for item in result.signed_peak_runs
    ] == [1, 1]
    assert all(len(item.execution_ids) == 1 for item in result.signed_peak_runs)
    assert all(
        item.algorithm_provenance.source_sha256
        == "07b7b101752219a95fa8ec83c561e89d7d0d5467cdfd4372b9d005828feb8931"
        for item in result.signed_peak_runs
    )
    assert result.signed_peak_runs[0].algorithm_provenance.scipy_version
    assert all(
        result.signed_peak_runs[index].parameter_snapshot.threshold
        == expected
        for index, expected in enumerate((-0.2, 0.2))
    )
    assert result.baseline_offset_by_channel is None
    assert result.random_seed == 5
    np.testing.assert_array_equal(captured[0][0], captured[0][1])
    assert any("biological validity" in item for item in result.warnings)
    assert any(
        item.name == "filtering_description" and item.status is MetadataStatus.UNKNOWN
        for item in result.metadata_snapshot
    )
    assert any(
        "Filtering information" in warning
        for prepared in approvals
        for warning in prepared.warnings
    )


def test_baseline_offset_is_a_recorded_transform_and_preserves_generated_signal(monkeypatch):
    import neuroscope.analysis.evaluation as evaluation_module

    scenario = _comparison_config(offset=(2.0,))
    generated_inputs = []
    original_generator = evaluation_module.generate_synthetic_recording

    def capture(config):
        generated = original_generator(config)
        generated_inputs.append((generated.recording.signal, generated.recording.signal.copy()))
        return generated

    monkeypatch.setattr(evaluation_module, "generate_synthetic_recording", capture)
    result = run_synthetic_detector_comparison(
        scenario, _comparison_registry(), approval_provider=_approve
    )

    assert result.baseline_offset_by_channel == (2.0,)
    assert any("baseline offset" in item for item in result.warnings)
    np.testing.assert_array_equal(generated_inputs[0][0], generated_inputs[0][1])


def test_comparison_records_confirmed_filtering_information():
    scenario = _comparison_config(filtering="No filter applied in this synthetic setup")
    result = run_synthetic_detector_comparison(
        scenario, _comparison_registry(), approval_provider=_approve
    )
    filtering = next(
        item for item in result.metadata_snapshot if item.name == "filtering_description"
    )
    assert filtering.status is MetadataStatus.CONFIRMED
    assert filtering.value == scenario.filtering_description
    assert all(
        "Filtering information" not in warning
        for run in (result.negative_detector, *result.signed_peak_runs)
        for warning in run.warnings
    )


@pytest.mark.parametrize("noise", [0.1, 0.3])
def test_comparison_preserves_noise_scenario_metadata(noise):
    scenario = _comparison_config(noise=noise, seed=int(noise * 10) + 9)
    result = run_synthetic_detector_comparison(
        scenario, _comparison_registry(), approval_provider=_approve
    )
    assert result.generator_config.noise_std_by_channel == (noise,)
    assert result.random_seed == scenario.generator_config.seed
    assert all(run.polarity_evaluation.matching_tolerance_samples == 0 for run in result.signed_peak_runs)
