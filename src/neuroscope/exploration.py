"""GUI-independent, read-only dataset summaries and sample selection."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .domain import DatasetMetadata, DatasetProfile


class MetadataStatus(str, Enum):
    """Evidence state for a metadata value."""

    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class MetadataValue:
    """A metadata value with its explicit evidence status."""

    value: object | None
    status: MetadataStatus


@dataclass(frozen=True, slots=True)
class SampleWindow:
    """Half-open sample interval ``[start, stop)`` in original array indices."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, Integral)
            or isinstance(self.stop, bool)
            or not isinstance(self.stop, Integral)
        ):
            raise TypeError("Sample window boundaries must be integers.")
        if self.start < 0:
            raise ValueError("Sample window start must be non-negative.")
        if self.stop <= self.start:
            raise ValueError("Sample window must be non-empty with stop greater than start.")


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Shape, axis interpretation, metadata status and confirmed-rate timing."""

    shape: tuple[int, ...]
    dtype: str
    channel_count: int
    sample_count: int
    channel_axis: int | None
    sample_axis: int
    metadata: Mapping[str, MetadataValue]

    @property
    def nominal_duration_s(self) -> float | None:
        """Return ``sample_count / sampling_rate_hz`` when rate is confirmed."""
        return self.metadata["nominal_duration_s"].value  # type: ignore[return-value]

    @property
    def final_sample_time_s(self) -> float | None:
        """Return ``(sample_count - 1) / sampling_rate_hz`` when rate is confirmed."""
        return self.metadata["final_sample_time_s"].value  # type: ignore[return-value]


class DatasetArraySource(Protocol):
    """Structural interface for a lazy numeric array with explicit close."""

    shape: tuple[int, ...]
    dtype: object

    def read_window(
        self,
        start: int,
        stop: int,
        *,
        channel_axis: int | None = None,
        channel_index: int | None = None,
    ) -> npt.NDArray: ...

    def close(self) -> None: ...


class DatasetExploration:
    """Read-only channel and sample-window access over an array or lazy source.

    A 2D array requires a profile whose channel axis was explicitly confirmed.
    A 1D array is treated as one channel. Returned arrays are read-only views
    for memory-backed inputs and bounded read-only selections for lazy sources.
    Original sample indexing is preserved; this model never transforms data.
    """

    _METADATA_FIELDS = (
        "sampling_rate_hz",
        "units",
        "signal_kind",
        "channel_axis",
        "time_axis",
        "notes",
        "filtering_description",
        "acquisition_origin",
    )

    def __init__(self, array: npt.NDArray | DatasetArraySource, profile: DatasetProfile) -> None:
        if not isinstance(profile, DatasetProfile):
            raise TypeError("profile must be a DatasetProfile.")
        if not isinstance(array, np.ndarray) and not all(
            hasattr(array, name) for name in ("shape", "dtype", "read_window", "close")
        ):
            raise TypeError("array must be a NumPy array or a closable lazy data source.")
        shape = tuple(array.shape)
        ndim = len(shape)
        if ndim not in (1, 2):
            raise ValueError(f"Only 1D or 2D arrays are supported (found {ndim}D).")
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("The dataset array must not be empty.")
        if not np.issubdtype(np.dtype(array.dtype), np.number):
            raise TypeError(f"The dataset array must be numeric (found {array.dtype}).")
        if tuple(profile.shape) != shape:
            raise ValueError("Dataset profile shape does not match the data source.")
        if profile.dtype != str(array.dtype):
            raise ValueError("Dataset profile dtype does not match the data source.")

        if ndim == 1:
            if profile.metadata.channel_axis is not None:
                raise ValueError("A 1D dataset must not declare a channel_axis.")
            channel_axis = None
            sample_axis = 0
            channel_count = 1
            sample_count = shape[0]
        else:
            channel_axis = profile.metadata.channel_axis
            if channel_axis not in (0, 1) or "channel_axis" not in profile.confirmed_fields:
                raise ValueError("A 2D dataset requires explicitly confirmed channel orientation.")
            sample_axis = 1 - channel_axis
            if profile.metadata.time_axis not in (None, sample_axis):
                raise ValueError("Dataset profile time_axis conflicts with channel orientation.")
            if "channel_axis" in profile.inferred_fields or "channel_axis" in profile.unknown_fields:
                raise ValueError("channel_axis cannot be both confirmed and inferred or unknown.")
            channel_count = shape[channel_axis]
            sample_count = shape[sample_axis]

        self._array = array
        self._dataset_id = profile.dataset_id
        self._channel_axis = channel_axis
        self._sample_axis = sample_axis
        self._channel_count = channel_count
        self._sample_count = sample_count
        self._summary = self._build_summary(array, profile)

    @classmethod
    def from_array(
        cls,
        array: npt.NDArray | DatasetArraySource,
        *,
        channel_axis: int | None = None,
        metadata: DatasetMetadata | None = None,
        confirmed_fields: tuple[str, ...] = (),
        inferred_fields: tuple[str, ...] = (),
        unknown_fields: tuple[str, ...] = (),
    ) -> DatasetExploration:
        """Build an exploration model, requiring orientation for 2D arrays."""
        if not isinstance(array, np.ndarray) and not hasattr(array, "shape"):
            raise TypeError("array must expose a numeric shape.")
        ndim = len(array.shape)
        if ndim == 2 and (
            isinstance(channel_axis, bool)
            or not isinstance(channel_axis, Integral)
            or channel_axis not in (0, 1)
        ):
            raise ValueError("Confirm channel_axis=0 or 1 before exploring a 2D array.")
        if ndim == 1 and channel_axis is not None:
            raise ValueError("A 1D array has one channel; channel_axis must be omitted.")
        if ndim not in (1, 2):
            raise ValueError(f"Only 1D or 2D arrays are supported (found {ndim}D).")

        if ndim == 2:
            channel_axis = int(channel_axis)
            time_axis = 1 - channel_axis
        else:
            time_axis = None
        if metadata is None:
            metadata = DatasetMetadata(channel_axis=channel_axis, time_axis=time_axis)
        elif ndim == 2:
            if metadata.channel_axis not in (None, channel_axis):
                raise ValueError("Metadata channel_axis conflicts with confirmed orientation.")
            if metadata.time_axis not in (None, time_axis):
                raise ValueError("Metadata time_axis conflicts with confirmed orientation.")
            metadata = metadata.model_copy(
                update={"channel_axis": channel_axis, "time_axis": time_axis}
            )

        path = Path("<in-memory>.npy")
        profile = DatasetProfile(
            dataset_id="in-memory",
            name="In-memory dataset",
            source_path=str(path),
            file_format="npy",
            shape=list(array.shape),
            dtype=str(array.dtype),
            metadata=metadata,
            confirmed_fields=list(confirmed_fields)
            + (["channel_axis"] if ndim == 2 and "channel_axis" not in confirmed_fields else []),
            inferred_fields=list(inferred_fields),
            unknown_fields=list(unknown_fields),
        )
        return cls(array, profile)

    @property
    def dataset_id(self) -> str:
        """Persistent dataset identity from the associated profile."""
        return self._dataset_id

    @property
    def summary(self) -> DatasetSummary:
        return self._summary

    @property
    def full_window(self) -> SampleWindow:
        """The full recording as a half-open interval ``[0, sample_count)``."""
        return SampleWindow(0, self._sample_count)

    def sample_time_seconds(self, sample_index: int) -> float:
        """Return a sample's relative time only for a confirmed sampling rate.

        No acquisition-time origin is added. Sample zero therefore maps to
        relative time zero, not to an asserted absolute acquisition timestamp.
        """
        self._validate_sample_index(sample_index)
        rate = self._summary.metadata["sampling_rate_hz"]
        if rate.status is not MetadataStatus.CONFIRMED or rate.value is None:
            raise ValueError("Seconds are unavailable unless sampling_rate_hz is confirmed.")
        return sample_index / float(rate.value)

    def channel(self, channel_index: int) -> npt.NDArray:
        """Return one complete channel; lazy sources materialize this channel."""
        self._validate_channel_index(channel_index)
        if not isinstance(self._array, np.ndarray):
            return self._array.read_window(
                0,
                self._sample_count,
                channel_axis=self._channel_axis,
                channel_index=channel_index,
            )
        if self._array.ndim == 1:
            selected = self._array
        elif self._channel_axis == 0:
            selected = self._array[channel_index, :]
        else:
            selected = self._array[:, channel_index]
        return self._readonly_view(selected)

    def window(
        self,
        window: SampleWindow,
        *,
        channel_index: int | None = None,
    ) -> npt.NDArray:
        """Return a read-only view of a validated half-open sample interval.

        Without ``channel_index``, a 2D result retains the original axis order.
        The stop index is exclusive and must not exceed the recording length.
        Empty windows and out-of-range boundaries are rejected.
        """
        if not isinstance(window, SampleWindow):
            raise TypeError("window must be a SampleWindow.")
        if window.stop > self._sample_count:
            raise ValueError("Sample window stop exceeds the recording length.")
        if channel_index is not None:
            self._validate_channel_index(channel_index)
            if not isinstance(self._array, np.ndarray):
                return self._array.read_window(
                    window.start,
                    window.stop,
                    channel_axis=self._channel_axis,
                    channel_index=channel_index,
                )
            return self.channel(channel_index)[window.start : window.stop]
        if not isinstance(self._array, np.ndarray):
            return self._array.read_window(
                window.start,
                window.stop,
                channel_axis=self._channel_axis,
            )
        if self._array.ndim == 1:
            selected = self._array[window.start : window.stop]
        elif self._sample_axis == 0:
            selected = self._array[window.start : window.stop, :]
        else:
            selected = self._array[:, window.start : window.stop]
        return self._readonly_view(selected)

    def read_sample(self, channel_index: int, sample_index: int) -> int | float:
        """Read one original sample without materializing a complete lazy channel."""
        self._validate_channel_index(channel_index)
        self._validate_sample_index(sample_index)
        if not isinstance(self._array, np.ndarray):
            value = self._array.read_window(
                sample_index,
                sample_index + 1,
                channel_axis=self._channel_axis,
                channel_index=channel_index,
            )[0]
        else:
            value = self.channel(channel_index)[sample_index]
        return value.item()

    def iter_channel_chunks(
        self,
        window: SampleWindow,
        *,
        channel_index: int,
        chunk_samples: int = 65_536,
    ) -> Iterator[tuple[int, npt.NDArray]]:
        """Yield bounded read-only chunks paired with original start indices."""
        if not isinstance(window, SampleWindow):
            raise TypeError("window must be a SampleWindow.")
        if window.stop > self._sample_count:
            raise ValueError("Sample window stop exceeds the recording length.")
        self._validate_channel_index(channel_index)
        if isinstance(chunk_samples, bool) or not isinstance(chunk_samples, Integral):
            raise TypeError("chunk_samples must be an integer.")
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive.")
        for start in range(window.start, window.stop, int(chunk_samples)):
            stop = min(start + int(chunk_samples), window.stop)
            yield start, self.window(
                SampleWindow(start, stop), channel_index=channel_index
            )

    def close(self) -> None:
        """Close a lazy source; in-memory NumPy datasets need no cleanup."""
        if not isinstance(self._array, np.ndarray):
            self._array.close()

    def _build_summary(
        self, array: npt.NDArray | DatasetArraySource, profile: DatasetProfile
    ) -> DatasetSummary:
        confirmed = set(profile.confirmed_fields)
        inferred = set(profile.inferred_fields)
        unknown = set(profile.unknown_fields)
        if confirmed & inferred or confirmed & unknown or inferred & unknown:
            raise ValueError("Metadata fields cannot have conflicting statuses.")

        raw_metadata = profile.metadata.model_dump()
        if len(self._array.shape) == 2 and raw_metadata["time_axis"] is None:
            raw_metadata["time_axis"] = self._sample_axis
        raw_metadata["acquisition_origin"] = None
        metadata_values: dict[str, MetadataValue] = {}
        for field in self._METADATA_FIELDS:
            value = raw_metadata[field]
            if field == "time_axis" and field not in confirmed | inferred | unknown:
                status = MetadataStatus.INFERRED if len(self._array.shape) == 2 else MetadataStatus.UNKNOWN
            elif field in confirmed:
                status = MetadataStatus.CONFIRMED
            elif field in inferred:
                status = MetadataStatus.INFERRED
            else:
                status = MetadataStatus.UNKNOWN
            if status is not MetadataStatus.UNKNOWN and value is None:
                raise ValueError(f"Metadata field {field!r} is marked {status.value} but has no value.")
            metadata_values[field] = MetadataValue(value, status)

        rate_entry = metadata_values["sampling_rate_hz"]
        if rate_entry.status is MetadataStatus.CONFIRMED and (
            not np.isfinite(float(rate_entry.value)) or float(rate_entry.value) <= 0
        ):
            raise ValueError("Confirmed sampling_rate_hz must be finite and positive.")
        if rate_entry.status is MetadataStatus.CONFIRMED:
            rate = float(rate_entry.value)
            nominal_duration = self._sample_count / rate
            final_sample_time = (self._sample_count - 1) / rate
            metadata_values["nominal_duration_s"] = MetadataValue(
                nominal_duration, MetadataStatus.INFERRED
            )
            metadata_values["final_sample_time_s"] = MetadataValue(
                final_sample_time, MetadataStatus.INFERRED
            )
        else:
            metadata_values["nominal_duration_s"] = MetadataValue(
                None, MetadataStatus.UNKNOWN
            )
            metadata_values["final_sample_time_s"] = MetadataValue(
                None, MetadataStatus.UNKNOWN
            )

        return DatasetSummary(
            shape=tuple(array.shape),
            dtype=str(array.dtype),
            channel_count=self._channel_count,
            sample_count=self._sample_count,
            channel_axis=self._channel_axis,
            sample_axis=self._sample_axis,
            metadata=MappingProxyType(metadata_values),
        )

    def _validate_channel_index(self, channel_index: int) -> None:
        if isinstance(channel_index, bool) or not isinstance(channel_index, Integral):
            raise TypeError("channel_index must be an integer.")
        channel_index = int(channel_index)
        if not 0 <= channel_index < self._channel_count:
            raise IndexError("channel_index is outside the dataset.")

    def _validate_sample_index(self, sample_index: int) -> None:
        if isinstance(sample_index, bool) or not isinstance(sample_index, Integral):
            raise TypeError("sample_index must be an integer.")
        sample_index = int(sample_index)
        if not 0 <= sample_index < self._sample_count:
            raise IndexError("sample_index is outside the dataset.")

    @staticmethod
    def _readonly_view(array: npt.NDArray) -> npt.NDArray:
        view = array.view()
        view.setflags(write=False)
        return view
