"""In-memory registration and controlled read-only access to loaded datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import numpy as np
import numpy.typing as npt

from .analysis.contracts import DataReference, DatasetReference, SampleInterval
from .exploration import DatasetExploration, MetadataValue, SampleWindow


class DatasetAccessError(ValueError):
    """A dataset handle or requested selection cannot be resolved safely."""


@dataclass(frozen=True, slots=True)
class DatasetRegistration:
    """Session-scoped handle plus persistent identity and known dimensions."""

    session_handle: UUID
    dataset: DatasetReference
    channel_count: int
    sample_count: int

    def reference(
        self,
        channels: tuple[int, ...],
        sample_interval: SampleInterval,
        *,
        component_id: str | None = None,
    ) -> DataReference:
        """Build a serializable selection; bounds are checked again at resolution."""
        return DataReference(
            dataset=self.dataset,
            session_handle=self.session_handle,
            channels=channels,
            sample_interval=sample_interval,
            component_id=component_id,
        )


@dataclass(frozen=True, slots=True)
class ResolvedChannelData:
    """Controlled channel access bounded to one approved original-sample interval.

    Accessing ``samples`` materializes that complete approved channel interval for
    existing array-based operations. ``read_window`` materializes only the
    requested subinterval and is intended for chunked scientific operations.
    """

    channel_index: int
    sample_interval: SampleInterval
    _exploration: DatasetExploration = field(repr=False, compare=False)
    _samples: npt.NDArray | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def samples(self) -> npt.NDArray:
        cached = self._samples
        if cached is None:
            cached = self._exploration.window(
                SampleWindow(self.sample_interval.start, self.sample_interval.stop),
                channel_index=self.channel_index,
            )
            object.__setattr__(self, "_samples", cached)
        return cached

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._exploration.summary.dtype)

    def read_window(self, start: int, stop: int) -> npt.NDArray:
        """Read a bounded half-open interval within this approved channel range."""
        if start < self.sample_interval.start or stop > self.sample_interval.stop:
            raise DatasetAccessError(
                "requested window is outside the approved DataReference interval."
            )
        return self._exploration.window(
            SampleWindow(start, stop), channel_index=self.channel_index
        )


@dataclass(frozen=True, slots=True)
class ResolvedSignalData:
    """Read-only views for exactly one requested channel/window selection."""

    reference: DataReference
    channels: tuple[ResolvedChannelData, ...]
    sample_interval: SampleInterval
    metadata: Mapping[str, MetadataValue]
    recording_sample_count: int

    def channel(self, channel_index: int) -> ResolvedChannelData:
        for channel in self.channels:
            if channel.channel_index == channel_index:
                return channel
        raise KeyError(f"Channel {channel_index} was not included in this data reference.")


@dataclass(slots=True)
class _RegisteredDataset:
    registration: DatasetRegistration
    exploration: DatasetExploration


class DatasetStore:
    """Session-local registry that resolves references through DatasetExploration.

    Registration retains the exploration's array or file-backed source.
    Resolution returns only requested read-only selections and does not
    preprocess or copy the full recording. NumPy write protection blocks ordinary
    writes, but it is not deep immutability for memory-backed datasets: another
    writable reference to the source may still mutate the same memory.
    """

    def __init__(self) -> None:
        self._datasets: dict[UUID, _RegisteredDataset] = {}

    def register(
        self,
        exploration: DatasetExploration,
        *,
        dataset: DatasetReference | None = None,
    ) -> DatasetRegistration:
        """Register an existing exploration without changing its underlying data."""
        if not isinstance(exploration, DatasetExploration):
            raise TypeError("exploration must be a DatasetExploration.")
        if dataset is None:
            dataset = DatasetReference(dataset_id=exploration.dataset_id)
        if not isinstance(dataset, DatasetReference):
            raise TypeError("dataset must be a DatasetReference.")
        if dataset.dataset_id != exploration.dataset_id:
            raise DatasetAccessError(
                "dataset identity does not match the registered exploration profile."
            )
        summary = exploration.summary
        handle = uuid4()
        registration = DatasetRegistration(
            session_handle=handle,
            dataset=dataset,
            channel_count=summary.channel_count,
            sample_count=summary.sample_count,
        )
        self._datasets[handle] = _RegisteredDataset(registration, exploration)
        return registration

    def unregister(self, session_handle: UUID | str) -> None:
        """Invalidate one session handle; its references never resolve again."""
        handle = self._parse_handle(session_handle)
        registered = self._datasets.pop(handle, None)
        if registered is None:
            raise DatasetAccessError(f"unknown or stale dataset handle: {handle}.")
        self._close_if_unregistered(registered.exploration)

    def registration_for_dataset(self, dataset: DatasetReference) -> DatasetRegistration:
        """Resolve a durable dataset identity to its unique current session handle.

        Pipelines persist ``DatasetReference`` values, never session handles. A
        missing or ambiguous active registration is rejected instead of guessing.
        """
        if not isinstance(dataset, DatasetReference):
            raise TypeError("dataset must be a DatasetReference.")
        matches = [
            item.registration
            for item in self._datasets.values()
            if item.registration.dataset == dataset
        ]
        if not matches:
            raise DatasetAccessError(
                f"dataset {dataset.dataset_id!r} has no active session registration."
            )
        if len(matches) > 1:
            raise DatasetAccessError(
                f"dataset {dataset.dataset_id!r} has ambiguous active registrations."
            )
        return matches[0]

    def replace(
        self,
        session_handle: UUID | str,
        exploration: DatasetExploration,
        *,
        dataset: DatasetReference | None = None,
    ) -> DatasetRegistration:
        """Atomically invalidate an old registration and return a fresh handle."""
        old_handle = self._parse_handle(session_handle)
        if old_handle not in self._datasets:
            raise DatasetAccessError(f"unknown or stale dataset handle: {old_handle}.")
        # Validate the replacement before invalidating the current registration.
        if not isinstance(exploration, DatasetExploration):
            raise TypeError("exploration must be a DatasetExploration.")
        if dataset is not None and dataset.dataset_id != exploration.dataset_id:
            raise DatasetAccessError(
                "dataset identity does not match the replacement exploration profile."
            )
        replacement = self.register(exploration, dataset=dataset)
        old = self._datasets.pop(old_handle)
        if old.exploration is not exploration:
            self._close_if_unregistered(old.exploration)
        return replacement

    def _close_if_unregistered(self, exploration: DatasetExploration) -> None:
        if not any(
            item.exploration is exploration for item in self._datasets.values()
        ):
            exploration.close()

    def resolve(self, reference: DataReference) -> ResolvedSignalData:
        """Resolve only requested channels and samples, preserving source indices."""
        if not isinstance(reference, DataReference):
            raise TypeError("reference must be a DataReference.")
        registered = self._datasets.get(reference.session_handle)
        if registered is None:
            raise DatasetAccessError(
                f"unknown or stale dataset handle: {reference.session_handle}."
            )
        registration = registered.registration
        if reference.dataset != registration.dataset:
            raise DatasetAccessError(
                "data reference persistent identity does not match this registration."
            )
        if reference.component_id is not None:
            raise DatasetAccessError(
                f"data component {reference.component_id!r} is not available for this dataset."
            )
        if reference.sample_interval.stop > registration.sample_count:
            raise DatasetAccessError(
                "sample interval stop exceeds the registered recording length."
            )
        try:
            SampleWindow(
                reference.sample_interval.start,
                reference.sample_interval.stop,
            )
            if any(
                channel_index >= registration.channel_count
                for channel_index in reference.channels
            ):
                raise IndexError("channel index is outside the registered dataset.")
            channels = tuple(
                ResolvedChannelData(
                    channel_index=channel_index,
                    sample_interval=reference.sample_interval,
                    _exploration=registered.exploration,
                )
                for channel_index in reference.channels
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise DatasetAccessError(f"invalid data selection: {exc}") from exc
        return ResolvedSignalData(
            reference=reference,
            channels=channels,
            sample_interval=reference.sample_interval,
            metadata=registered.exploration.summary.metadata,
            recording_sample_count=registration.sample_count,
        )

    @staticmethod
    def _parse_handle(handle: UUID | str) -> UUID:
        try:
            return handle if isinstance(handle, UUID) else UUID(handle)
        except (TypeError, ValueError, AttributeError) as exc:
            raise DatasetAccessError(f"invalid dataset session handle: {handle!r}.") from exc
