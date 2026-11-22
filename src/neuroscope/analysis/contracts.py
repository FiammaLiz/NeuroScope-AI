"""Typed, GUI-independent definitions for scientific analysis operations."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from inspect import signature
from typing import Any, Generic, Protocol, TypeVar
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    field_validator,
    model_validator,
)

from ..exploration import MetadataStatus


class OperationModel(BaseModel):
    """Base for operation-specific schemas; models reject extras and assignment."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetReference(OperationModel):
    """Identity/reference only; this is not proof that contents are unchanged."""

    dataset_id: str = Field(min_length=1)
    source_reference: str | None = Field(default=None, min_length=1)
    content_fingerprint: str | None = Field(default=None, min_length=1)


class DataReference(OperationModel):
    """Serializable selection; it identifies data but never contains array values.

    Channels use zero-based indices in the profile's confirmed channel orientation.
    The sample interval is zero-based and half-open: ``[start, stop)``.
    ``session_handle`` identifies one in-memory registration, not a persistent file.
    """

    dataset: DatasetReference
    session_handle: UUID
    channels: tuple[StrictInt, ...] = Field(min_length=1)
    sample_interval: SampleInterval
    component_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def valid_channel_selection(self) -> DataReference:
        if any(channel < 0 for channel in self.channels):
            raise ValueError("channel indices must be non-negative.")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channel indices must be unique.")
        return self


class ResolvedChannelAccess(Protocol):
    """One selected channel's read-only samples in original sample coordinates."""

    channel_index: int
    samples: Any


class ResolvedDataAccess(Protocol):
    """Narrow access to only the channels and interval named by one reference."""

    reference: DataReference
    channels: tuple[ResolvedChannelAccess, ...]
    sample_interval: SampleInterval
    metadata: Mapping[str, Any]

    def channel(self, channel_index: int) -> ResolvedChannelAccess: ...


class OperationRuntimeAccess(Protocol):
    """Execution-only access; never serialized into prepared requests or records."""

    resolved_data: tuple[ResolvedDataAccess, ...]
    artifact_store: Any | None
    isi_artifact_store: Any | None
    mua_artifact_store: Any | None
    execution_id: str
    started_at_utc: Any
    approval_id: str
    approved_by: str

    def data_for(self, reference: DataReference) -> ResolvedDataAccess: ...


class MetadataSnapshot(OperationModel):
    """A metadata value and its evidence state preserved for one execution."""

    name: str = Field(min_length=1)
    value: JsonValue = None
    status: MetadataStatus


class SampleInterval(OperationModel):
    """Half-open interval in original sample coordinates: ``[start, stop)``."""

    start: int = Field(ge=0, strict=True)
    stop: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def non_empty(self) -> SampleInterval:
        if self.stop <= self.start:
            raise ValueError("sample interval must be non-empty (stop > start).")
        return self


class ExecutionStatus(str, Enum):
    """Successful or failed operation execution state."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PrerequisiteReference(OperationModel):
    """Reference to a completed operation output required by another operation."""

    execution_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    operation_version: str = Field(min_length=1)
    output_reference: str = Field(min_length=1)
    status: ExecutionStatus = ExecutionStatus.SUCCEEDED

    @field_validator("operation_id")
    @classmethod
    def namespaced_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:[.-][a-z0-9]+)+", value):
            raise ValueError("prerequisite operation_id must be namespaced.")
        return value

    @field_validator("operation_version")
    @classmethod
    def semantic_version(cls, value: str) -> str:
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", value):
            raise ValueError("prerequisite operation_version must be semantic version.")
        return value


class ReproducibilityInfo(OperationModel):
    """Optional operation-specific seed and configuration references."""

    seed: int | None = Field(default=None, ge=0, strict=True)
    rng_algorithm: str | None = None
    configuration_reference: str | None = None


class OperationContext(OperationModel):
    """Shared execution context; operation-specific payloads remain typed separately."""

    dataset_reference: DatasetReference | None = None
    metadata: tuple[MetadataSnapshot, ...] = ()
    selected_channels: tuple[StrictInt, ...] = ()
    sample_interval: SampleInterval | None = None
    prerequisites: tuple[PrerequisiteReference, ...] = ()
    reproducibility: ReproducibilityInfo | None = None

    @model_validator(mode="after")
    def unique_context_entries(self) -> OperationContext:
        metadata_names = [item.name for item in self.metadata]
        if len(set(metadata_names)) != len(metadata_names):
            raise ValueError("metadata snapshot names must be unique.")
        if any(index < 0 for index in self.selected_channels):
            raise ValueError("selected channel indices must be non-negative.")
        if len(set(self.selected_channels)) != len(self.selected_channels):
            raise ValueError("selected channel indices must be unique.")
        prerequisite_ids = [item.operation_id for item in self.prerequisites]
        if len(set(prerequisite_ids)) != len(prerequisite_ids):
            raise ValueError("prerequisite operation IDs must be unique.")
        return self


class MetadataRequirement(OperationModel):
    """A field requirement; optional unknown values remain present as UNKNOWN."""

    name: str = Field(min_length=1)
    required: bool = True
    require_confirmed: bool = False

    @model_validator(mode="after")
    def confirmed_must_be_required(self) -> MetadataRequirement:
        if self.require_confirmed and not self.required:
            raise ValueError("a confirmed metadata requirement must also be required.")
        return self


InputT = TypeVar("InputT", bound=OperationModel)
ParametersT = TypeVar("ParametersT", bound=OperationModel)
OutputT = TypeVar("OutputT", bound=OperationModel)


@dataclass(frozen=True, slots=True)
class OperationResult(Generic[OutputT]):
    """Typed scientific output plus output metadata and run-time warnings."""

    output: OutputT
    output_metadata: tuple[MetadataSnapshot, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.output, OperationModel):
            raise TypeError("operation output must be an OperationModel.")
        if any(not isinstance(item, MetadataSnapshot) for item in self.output_metadata):
            raise TypeError("output_metadata must contain MetadataSnapshot values.")
        if any(not isinstance(warning, str) or not warning for warning in self.warnings):
            raise ValueError("operation warnings must be non-empty strings.")


@dataclass(frozen=True, slots=True)
class OperationDefinition(Generic[InputT, ParametersT, OutputT]):
    """Versioned contract and implementation for one reusable operation."""

    operation_id: str
    name: str
    description: str
    version: str
    implementation_version: str
    input_model: type[InputT]
    parameters_model: type[ParametersT]
    output_model: type[OutputT]
    execute: Callable[..., OperationResult[OutputT] | OutputT]
    metadata_requirements: tuple[MetadataRequirement, ...] = ()
    prerequisites: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    prepare_warnings: Callable[[InputT, ParametersT, OperationContext], tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str):
            raise TypeError("operation_id must be a string.")
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:[.-][a-z0-9]+)+", self.operation_id):
            raise ValueError("operation_id must be a stable, namespaced lowercase ID.")
        if (
            not isinstance(self.name, str)
            or not isinstance(self.description, str)
            or not self.name.strip()
            or not self.description.strip()
        ):
            raise ValueError("operation name and description must be non-empty.")
        if not isinstance(self.version, str) or not re.fullmatch(
            r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", self.version
        ):
            raise ValueError("operation version must use semantic version syntax.")
        if not isinstance(self.implementation_version, str) or not self.implementation_version.strip():
            raise ValueError("implementation_version must be non-empty.")
        for label, model in (
            ("input_model", self.input_model),
            ("parameters_model", self.parameters_model),
            ("output_model", self.output_model),
        ):
            if not isinstance(model, type) or not issubclass(model, OperationModel):
                raise TypeError(f"{label} must derive from OperationModel.")
        if not callable(self.execute):
            raise TypeError("execute must be a callable operation implementation.")
        try:
            callback_signature = signature(self.execute)
        except (TypeError, ValueError) as exc:
            raise TypeError("execute must have an inspectable callable signature.") from exc
        try:
            callback_signature.bind(None, None, OperationContext(), None)
        except TypeError:
            try:
                callback_signature.bind(None, None, OperationContext())
            except TypeError as exc:
                raise TypeError(
                    "execute must accept input, parameters, and operation context, "
                    "with optional runtime access as a fourth argument."
                ) from exc
        if not isinstance(self.metadata_requirements, tuple) or any(
            not isinstance(item, MetadataRequirement)
            for item in self.metadata_requirements
        ):
            raise TypeError("metadata_requirements must be a tuple of requirements.")
        requirement_names = [item.name for item in self.metadata_requirements]
        if len(set(requirement_names)) != len(requirement_names):
            raise ValueError("metadata requirement names must be unique.")
        if not isinstance(self.prerequisites, tuple) or any(
            not isinstance(item, str) or not item for item in self.prerequisites
        ):
            raise TypeError("prerequisites must be a tuple of operation IDs.")
        if any(
            not re.fullmatch(r"[a-z][a-z0-9]*(?:[.-][a-z0-9]+)+", item)
            for item in self.prerequisites
        ):
            raise ValueError("prerequisite IDs must be stable, namespaced lowercase IDs.")
        if len(set(self.prerequisites)) != len(self.prerequisites):
            raise ValueError("prerequisite operation IDs must be unique.")
        if self.prepare_warnings is not None:
            if not callable(self.prepare_warnings):
                raise TypeError("prepare_warnings must be callable when provided.")
            try:
                signature(self.prepare_warnings).bind(None, None, OperationContext())
            except TypeError as exc:
                raise TypeError(
                    "prepare_warnings must accept input, parameters, and context."
                ) from exc
        for field_name, values in (
            ("warnings", self.warnings),
            ("limitations", self.limitations),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise TypeError(f"{field_name} must be a tuple of non-empty strings.")
