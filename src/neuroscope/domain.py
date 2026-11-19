from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    sampling_rate_hz: float | None = Field(default=None, gt=0)
    units: str | None = None
    signal_kind: str | None = None  # e.g. raw_voltage, spike_times, binned_activity
    channel_axis: int | None = None
    time_axis: int | None = None
    notes: str | None = None


class DatasetProfile(BaseModel):
    dataset_id: str
    name: str
    source_path: str
    file_format: str
    shape: list[int]
    dtype: str
    metadata: DatasetMetadata = Field(default_factory=DatasetMetadata)
    confirmed_fields: list[str] = Field(default_factory=list)
    inferred_fields: list[str] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=lambda: ["sampling_rate_hz", "units", "signal_kind"])
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_array(cls, path: Path, array: Any, *, variable_name: str | None = None,
                   channel_axis: int | None = None):
        import hashlib

        if array.ndim not in (1, 2):
            raise ValueError("Only 1D or 2D arrays can be profiled.")
        if array.ndim == 2 and channel_axis not in (0, 1):
            raise ValueError("Confirm channel_axis=0 or 1 before profiling a 2D array.")
        if array.ndim == 1 and channel_axis is not None:
            raise ValueError("A 1D array has one channel; channel_axis must be omitted.")

        resolved = str(path.resolve())
        dataset_id = hashlib.sha256(resolved.encode()).hexdigest()[:12]
        name = path.stem if variable_name is None else f"{path.stem}:{variable_name}"
        return cls(
            dataset_id=dataset_id,
            name=name,
            source_path=resolved,
            file_format=path.suffix.lower().lstrip("."),
            shape=list(array.shape),
            dtype=str(array.dtype),
            metadata=DatasetMetadata(channel_axis=channel_axis,
                                     time_axis=(1 - channel_axis if channel_axis is not None else None)),
            confirmed_fields=["channel_axis"] if array.ndim == 2 else [],
        )
