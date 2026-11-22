"""Signed threshold peak detector based on Fiamma Liz Leites' reference code.

Leites authored the original ``spike_detection.py`` implementation. This is a
separate NeuroScope operation based on that method; its source-method
provenance does not imply collaborator authorship or endorsement.
"""

from __future__ import annotations

from numbers import Real
from typing import Literal

import numpy as np
import scipy
from pydantic import Field, field_validator, model_validator
from scipy.signal import find_peaks

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

OPERATION_ID = "neuroscope.spike.detect-signed-peaks"
OPERATION_VERSION = "1.0.0"
SOURCE_SHA256 = "07b7b101752219a95fa8ec83c561e89d7d0d5467cdfd4372b9d005828feb8931"
IMPLEMENTATION_VERSION = f"owner-source-sha256-{SOURCE_SHA256}-adapter-v1"

NOISE_ESTIMATOR_DIVISOR = 0.6745
PROMINENCE_FACTOR = 0.75
MAXIMUM_AMPLITUDE_FACTOR = 25.0

FILTERING_WARNING = (
    "Filtering information is missing, unknown, inferred, or unconfirmed. "
    "No filtering is assumed or applied; review signal suitability before interpreting results."
)


class SignedPeakInput(OperationModel):
    """One selected continuous channel and a half-open original sample interval."""

    signal: DataReference

    @model_validator(mode="after")
    def one_channel_only(self) -> SignedPeakInput:
        if len(self.signal.channels) != 1:
            raise ValueError("signed peak detection requires exactly one channel.")
        return self


class SignedPeakParameters(OperationModel):
    """Explicit absolute-threshold or source-compatible noise-multiplier mode."""

    threshold_mode: Literal["absolute", "noise_multiplier"]
    threshold: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Signed absolute threshold in the signal's sample-amplitude units.",
    )
    stdmin: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Signed multiplier applied to the uncentered noise estimate.",
    )
    noise_override: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Absolute-mode noise value used only by the strict amplitude cap.",
    )

    @field_validator("threshold", "stdmin", "noise_override", mode="before")
    @classmethod
    def finite_real_values(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("threshold, stdmin, and noise_override must be real numbers.")
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError("threshold, stdmin, and noise_override must be finite.")
        return converted

    @model_validator(mode="after")
    def mode_fields_are_unambiguous(self) -> SignedPeakParameters:
        if self.threshold_mode == "absolute":
            if self.threshold is None or self.stdmin is not None:
                raise ValueError(
                    "absolute mode requires threshold and forbids stdmin."
                )
        elif self.stdmin is None or self.threshold is not None or self.noise_override is not None:
            raise ValueError(
                "noise_multiplier mode requires stdmin and forbids threshold and noise_override."
            )
        return self

    @property
    def polarity(self) -> Literal["positive", "negative", "none"]:
        signed_value = self.threshold if self.threshold_mode == "absolute" else self.stdmin
        if signed_value == 0:
            return "none"
        return "positive" if signed_value is not None and signed_value > 0 else "negative"


class SciPyPeakProperties(OperationModel):
    """SciPy properties aligned with retained detections; base indices are window-relative."""

    peak_heights: tuple[float, ...] | None = None
    prominences: tuple[float, ...] | None = None
    left_bases: tuple[int, ...] | None = None
    right_bases: tuple[int, ...] | None = None


class SignedPeakProvenance(OperationModel):
    source_sha256: str
    noise_estimator_divisor: float
    prominence_factor: float
    maximum_amplitude_factor: float
    numpy_version: str
    scipy_version: str
    filtering_description: str | None
    filtering_status: MetadataStatus
    filtering_warning: bool


class SignedPeakOutput(OperationModel):
    """Serialized detector results with explicit local/original coordinates."""

    window_relative_sample_indices: tuple[int, ...]
    original_sample_indices: tuple[int, ...]
    times_s_relative_to_window: tuple[float, ...] | None
    time_reference: Literal["window_start_sample_zero"] = "window_start_sample_zero"
    amplitudes: tuple[float, ...]
    noise_estimate: float = Field(ge=0, allow_inf_nan=False)
    noise_source: Literal["estimated", "overridden"]
    threshold: float = Field(allow_inf_nan=False)
    threshold_mode: Literal["absolute", "noise_multiplier"]
    polarity: Literal["positive", "negative", "none"]
    channel_index: int = Field(ge=0, strict=True)
    sample_interval: SampleInterval
    peak_properties: SciPyPeakProperties
    peak_property_index_reference: Literal["window_relative_samples"] = "window_relative_samples"
    peak_properties_signal: Literal["original", "inverted", "none"]
    sampling_rate_hz: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    sampling_rate_status: MetadataStatus
    amplitude_units: str | None = None
    amplitude_units_status: MetadataStatus
    filtering_description: str | None = None
    filtering_status: MetadataStatus
    algorithm_provenance: SignedPeakProvenance


def estimate_signed_peak_noise(signal: np.ndarray) -> float:
    """Return the source estimator: median absolute amplitude divided by 0.6745."""
    values = np.asarray(signal, dtype=float)
    return float(np.median(np.abs(values)) / NOISE_ESTIMATOR_DIVISOR)


def threshold_from_signed_peak_noise(noise: float, stdmin: float) -> float:
    """Apply the source's signed multiplier convention."""
    return float(stdmin) * float(noise)


def _detect_signed_peaks(
    signal: np.ndarray,
    threshold: float,
    noise: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], float]:
    """Faithful core of the audited source, after the adapter's safety checks."""
    values = np.asarray(signal, dtype=float)
    if noise is None:
        noise = estimate_signed_peak_noise(values)
    threshold = float(threshold)
    if threshold == 0:
        return np.array([], dtype=int), np.array([], dtype=float), {}, float(noise)

    threshold_abs = abs(threshold)
    prominence = threshold_abs * PROMINENCE_FACTOR
    max_amplitude = MAXIMUM_AMPLITUDE_FACTOR * noise
    detection_signal = -values if threshold < 0 else values
    peaks, properties = find_peaks(
        detection_signal,
        height=threshold_abs,
        prominence=prominence,
    )
    amplitudes = values[peaks]
    valid = np.abs(amplitudes) < max_amplitude
    peaks = peaks[valid]
    amplitudes = amplitudes[valid]
    filtered_properties: dict[str, np.ndarray] = {}
    for key, value in properties.items():
        property_values = np.asarray(value)
        filtered_properties[key] = (
            property_values[valid]
            if len(property_values) == len(valid)
            else property_values
        )
    return peaks, amplitudes, filtered_properties, float(noise)


def _prepare_filtering_warnings(
    input_data: SignedPeakInput,
    parameters: SignedPeakParameters,
    context: OperationContext,
) -> tuple[str, ...]:
    del input_data, parameters
    snapshot = next(
        (item for item in context.metadata if item.name == "filtering_description"),
        None,
    )
    if (
        snapshot is None
        or snapshot.status is not MetadataStatus.CONFIRMED
        or snapshot.value is None
        or not str(snapshot.value).strip()
    ):
        return (FILTERING_WARNING,)
    return ()


def _execute_signed_peak_detection(
    input_data: SignedPeakInput,
    parameters: SignedPeakParameters,
    context: OperationContext,
    runtime: OperationRuntimeAccess,
) -> OperationResult[SignedPeakOutput]:
    reference = input_data.signal
    resolved = runtime.data_for(reference)
    if len(resolved.channels) != 1:
        raise ValueError("runtime data must contain exactly the referenced channel.")
    channel = reference.channels[0]
    samples = resolved.channel(channel).samples
    if samples.ndim != 1:
        raise ValueError("signed peak detection requires a one-dimensional channel view.")
    if (
        not np.issubdtype(samples.dtype, np.number)
        or np.issubdtype(samples.dtype, np.complexfloating)
        or samples.size == 0
        or not np.isfinite(samples).all()
    ):
        raise ValueError("signal must be a non-empty finite real numeric array.")
    values = np.asarray(samples, dtype=float)

    actual_metadata = dict(resolved.metadata)
    context_metadata = {item.name: item for item in context.metadata}
    fields = ("signal_kind", "sampling_rate_hz", "units", "filtering_description")
    metadata: dict[str, MetadataValue] = {}
    for name in fields:
        actual = actual_metadata[name]
        requested = context_metadata.get(name)
        # Filtering is explicitly advisory: a missing snapshot warns at prepare
        # time, then the registered metadata is preserved in the result.
        if requested is None and name == "filtering_description":
            metadata[name] = actual
            continue
        if requested is None:
            if actual.status is MetadataStatus.UNKNOWN and actual.value is None:
                metadata[name] = actual
                continue
            raise ValueError(
                f"execution context must snapshot dataset metadata {name!r} and its status."
            )
        if requested.status is not actual.status or requested.value != actual.value:
            raise ValueError(
                f"execution context metadata {name!r} does not match the registered dataset."
            )
        metadata[name] = actual

    if parameters.threshold_mode == "absolute":
        threshold = float(parameters.threshold)
        noise_override = parameters.noise_override
        peaks, amplitudes, properties, noise = _detect_signed_peaks(
            values, threshold, noise_override
        )
        noise_source: Literal["estimated", "overridden"] = (
            "overridden" if noise_override is not None else "estimated"
        )
    else:
        noise = estimate_signed_peak_noise(values)
        threshold = threshold_from_signed_peak_noise(noise, float(parameters.stdmin))
        peaks, amplitudes, properties, noise = _detect_signed_peaks(
            values, threshold, noise
        )
        noise_source = "estimated"

    properties_model = SciPyPeakProperties(
        peak_heights=(
            tuple(float(value) for value in properties["peak_heights"])
            if "peak_heights" in properties
            else None
        ),
        prominences=(
            tuple(float(value) for value in properties["prominences"])
            if "prominences" in properties
            else None
        ),
        left_bases=(
            tuple(int(value) for value in properties["left_bases"])
            if "left_bases" in properties
            else None
        ),
        right_bases=(
            tuple(int(value) for value in properties["right_bases"])
            if "right_bases" in properties
            else None
        ),
    )
    window_indices = tuple(int(value) for value in peaks)
    rate_value = metadata["sampling_rate_hz"]
    confirmed_rate = (
        float(rate_value.value)
        if rate_value.status is MetadataStatus.CONFIRMED and rate_value.value is not None
        else None
    )
    filter_value = metadata["filtering_description"]
    filtering_warning = (
        filter_value.status is not MetadataStatus.CONFIRMED
        or filter_value.value is None
        or not str(filter_value.value).strip()
        or "filtering_description" not in context_metadata
    )
    units = metadata["units"]
    output = SignedPeakOutput(
        window_relative_sample_indices=window_indices,
        original_sample_indices=tuple(reference.sample_interval.start + item for item in window_indices),
        times_s_relative_to_window=(
            tuple(item / confirmed_rate for item in window_indices)
            if confirmed_rate is not None
            else None
        ),
        amplitudes=tuple(float(value) for value in amplitudes),
        noise_estimate=noise,
        noise_source=noise_source,
        threshold=threshold,
        threshold_mode=parameters.threshold_mode,
        polarity=parameters.polarity,
        channel_index=channel,
        sample_interval=reference.sample_interval,
        peak_properties=properties_model,
        peak_properties_signal=(
            "none"
            if parameters.polarity == "none"
            else "inverted"
            if parameters.polarity == "negative"
            else "original"
        ),
        sampling_rate_hz=(
            float(rate_value.value) if rate_value.value is not None else None
        ),
        sampling_rate_status=rate_value.status,
        amplitude_units=str(units.value) if units.value is not None else None,
        amplitude_units_status=units.status,
        filtering_description=(
            str(filter_value.value) if filter_value.value is not None else None
        ),
        filtering_status=filter_value.status,
        algorithm_provenance=SignedPeakProvenance(
            source_sha256=SOURCE_SHA256,
            noise_estimator_divisor=NOISE_ESTIMATOR_DIVISOR,
            prominence_factor=PROMINENCE_FACTOR,
            maximum_amplitude_factor=MAXIMUM_AMPLITUDE_FACTOR,
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            filtering_description=(
                str(filter_value.value) if filter_value.value is not None else None
            ),
            filtering_status=filter_value.status,
            filtering_warning=filtering_warning,
        ),
    )
    output_metadata = tuple(
        MetadataSnapshot(name=name, value=value.value, status=value.status)
        for name, value in metadata.items()
    )
    return OperationResult(
        output=output,
        output_metadata=output_metadata,
        warnings=(FILTERING_WARNING,) if filtering_warning else (),
    )


SIGNED_PEAK_DETECTOR = OperationDefinition(
    operation_id=OPERATION_ID,
    name="Signed threshold peak detection",
    description=(
        "Detect positive or negative peaks using the owner-audited uncentered "
        "median-absolute-amplitude estimator, SciPy prominence, and strict amplitude cap."
    ),
    version=OPERATION_VERSION,
    implementation_version=IMPLEMENTATION_VERSION,
    input_model=SignedPeakInput,
    parameters_model=SignedPeakParameters,
    output_model=SignedPeakOutput,
    execute=_execute_signed_peak_detection,
    metadata_requirements=(
        MetadataRequirement(name="signal_kind", required=False),
        MetadataRequirement(name="sampling_rate_hz", required=False),
        MetadataRequirement(name="units", required=False),
        MetadataRequirement(name="filtering_description", required=False),
    ),
    warnings=(
        "The reference describes filtered neural signals; no filter is applied or required by this operation.",
        "The noise estimator uses median absolute amplitude without median-centering.",
    ),
    limitations=(
        "Threshold and prominence use the original signal amplitude scale relative to zero.",
        "No refractory-period enforcement, waveform validation, or artifact rejection is performed.",
        "SciPy find_peaks endpoint and plateau behavior is preserved.",
        "Synthetic performance does not establish biological validity.",
        "Times are relative to the selected window's sample zero, not an acquisition-time origin.",
    ),
    prepare_warnings=_prepare_filtering_warnings,
)


def register_signed_peak_detector(
    registry: OperationRegistry,
) -> OperationDefinition[SignedPeakInput, SignedPeakParameters, SignedPeakOutput]:
    """Register the owner-authored method adapter in the existing registry."""
    if not isinstance(registry, OperationRegistry):
        raise TypeError("registry must be an OperationRegistry.")
    registry.register(SIGNED_PEAK_DETECTOR)
    return SIGNED_PEAK_DETECTOR
