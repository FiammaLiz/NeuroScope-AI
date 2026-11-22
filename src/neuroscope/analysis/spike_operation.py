"""Operation adapter for the existing negative-going spike detector."""

from __future__ import annotations

from numbers import Real
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ..exploration import MetadataStatus, MetadataValue
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
from .registry import OperationRegistry
from .spikes import detect_negative_spikes

OPERATION_ID = "neuroscope.spike.detect-negative"
OPERATION_VERSION = "1.0.0"
IMPLEMENTATION_VERSION = "repo-mad-negative-minima-v1"


class NegativeSpikeInput(OperationModel):
    """One continuous signal channel and original-recording sample interval."""

    signal: DataReference

    @model_validator(mode="after")
    def one_channel_only(self) -> NegativeSpikeInput:
        if len(self.signal.channels) != 1:
            raise ValueError("negative spike detection requires exactly one channel.")
        return self


class NegativeSpikeParameters(OperationModel):
    """Existing detector parameters; threshold_sd retains its -4.0 default."""

    threshold_sd: float = Field(default=-4.0, lt=0, allow_inf_nan=False)

    @field_validator("threshold_sd", mode="before")
    @classmethod
    def numeric_threshold(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("threshold_sd must be a finite negative real number.")
        return float(value)


class NegativeSpikeOutput(OperationModel):
    """Detector output with explicit indices, times, and measurement status.

    Amplitudes, noise scale, and threshold all use ``amplitude_units`` and its
    evidence status. Times, when present, are relative to the selected window.
    """

    window_relative_sample_indices: tuple[int, ...]
    original_sample_indices: tuple[int, ...]
    times_s_relative_to_window: tuple[float, ...] | None
    time_reference: Literal["window_start_sample_zero"] = "window_start_sample_zero"
    amplitudes: tuple[float, ...]
    noise_sigma: float = Field(allow_inf_nan=False)
    threshold: float = Field(allow_inf_nan=False)
    threshold_sd: float = Field(lt=0, allow_inf_nan=False)
    channel_index: int = Field(ge=0, strict=True)
    sample_interval: SampleInterval
    sampling_rate_hz: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    sampling_rate_status: MetadataStatus
    amplitude_units: str | None = None
    amplitude_units_status: MetadataStatus


_METADATA_FIELDS = ("signal_kind", "sampling_rate_hz", "units")


def _check_metadata_snapshot(
    name: str,
    runtime_metadata: dict[str, MetadataValue],
    context_metadata: dict[str, MetadataSnapshot],
) -> MetadataValue:
    actual = runtime_metadata[name]
    requested = context_metadata.get(name)
    if requested is None:
        if actual.status is MetadataStatus.UNKNOWN and actual.value is None:
            return actual
        raise ValueError(
            f"execution context must snapshot dataset metadata {name!r} and its status."
        )
    if requested.status is not actual.status or requested.value != actual.value:
        raise ValueError(
            f"execution context metadata {name!r} does not match the registered dataset."
        )
    return actual


def _execute_negative_spike_detection(
    input_data: NegativeSpikeInput,
    parameters: NegativeSpikeParameters,
    context: OperationContext,
    runtime: OperationRuntimeAccess,
) -> OperationResult[NegativeSpikeOutput]:
    reference = input_data.signal
    if len(reference.channels) != 1:
        raise ValueError("negative spike detection requires exactly one channel.")
    resolved = runtime.data_for(reference)
    if len(resolved.channels) != 1:
        raise ValueError("runtime data must contain exactly the referenced channel.")
    channel = reference.channels[0]
    samples = resolved.channel(channel).samples
    if samples.ndim != 1:
        raise ValueError("negative spike detection requires a one-dimensional channel view.")

    runtime_metadata = dict(resolved.metadata)
    context_metadata = {item.name: item for item in context.metadata}
    metadata = {
        name: _check_metadata_snapshot(name, runtime_metadata, context_metadata)
        for name in _METADATA_FIELDS
    }
    signal_kind = metadata["signal_kind"]
    if (
        signal_kind.status is not MetadataStatus.CONFIRMED
        or signal_kind.value != "raw_voltage"
    ):
        raise ValueError(
            "negative spike detection requires confirmed signal_kind='raw_voltage'."
        )

    sampling_rate = metadata["sampling_rate_hz"]
    confirmed_rate = (
        float(sampling_rate.value)
        if sampling_rate.status is MetadataStatus.CONFIRMED
        and sampling_rate.value is not None
        else None
    )
    detector_result = detect_negative_spikes(
        samples,
        sampling_rate_hz=confirmed_rate,
        threshold_sd=parameters.threshold_sd,
    )
    window_samples = tuple(int(value) for value in detector_result["samples"])
    start = reference.sample_interval.start
    output = NegativeSpikeOutput(
        window_relative_sample_indices=window_samples,
        original_sample_indices=tuple(start + sample for sample in window_samples),
        times_s_relative_to_window=(
            None
            if detector_result["times_s"] is None
            else tuple(float(value) for value in detector_result["times_s"])
        ),
        amplitudes=tuple(float(value) for value in detector_result["amplitudes"]),
        noise_sigma=float(detector_result["noise_sigma"]),
        threshold=float(detector_result["threshold"]),
        threshold_sd=float(detector_result["threshold_sd"]),
        channel_index=channel,
        sample_interval=reference.sample_interval,
        sampling_rate_hz=(
            float(sampling_rate.value)
            if sampling_rate.value is not None
            else None
        ),
        sampling_rate_status=sampling_rate.status,
        amplitude_units=(
            str(metadata["units"].value)
            if metadata["units"].value is not None
            else None
        ),
        amplitude_units_status=metadata["units"].status,
    )
    output_metadata = tuple(
        MetadataSnapshot(name=name, value=value.value, status=value.status)
        for name, value in metadata.items()
    )
    return OperationResult(
        output=output,
        output_metadata=output_metadata,
    )


NEGATIVE_SPIKE_DETECTOR = OperationDefinition(
    operation_id=OPERATION_ID,
    name="Negative-going spike detection",
    description=(
        "Detect interior negative local minima in one confirmed continuous raw-voltage channel "
        "using the repository's median-centered MAD threshold."
    ),
    version=OPERATION_VERSION,
    implementation_version=IMPLEMENTATION_VERSION,
    input_model=NegativeSpikeInput,
    parameters_model=NegativeSpikeParameters,
    output_model=NegativeSpikeOutput,
    execute=_execute_negative_spike_detection,
    metadata_requirements=(
        MetadataRequirement(
            name="signal_kind", required=True, require_confirmed=True
        ),
        MetadataRequirement(name="sampling_rate_hz", required=False),
        MetadataRequirement(name="units", required=False),
    ),
    warnings=(
        "Detects negative-going events only; positive-polarity detection is unsupported.",
        "Noise scale is median-centered MAD divided by 0.6744897501960817.",
        "Thresholding uses raw sample values without baseline subtraction.",
    ),
    limitations=(
        "No refractory-period enforcement.",
        "No prominence criterion.",
        "No waveform validation or artifact rejection.",
        "Synthetic performance does not establish biological validity.",
        "Times are relative to the selected window's sample zero, not an acquisition-time origin.",
    ),
)


def register_negative_spike_detector(
    registry: OperationRegistry,
) -> OperationDefinition[NegativeSpikeInput, NegativeSpikeParameters, NegativeSpikeOutput]:
    """Register the production adapter once in the existing operation registry."""
    if not isinstance(registry, OperationRegistry):
        raise TypeError("registry must be an OperationRegistry.")
    registry.register(NEGATIVE_SPIKE_DETECTOR)
    return NEGATIVE_SPIKE_DETECTOR
