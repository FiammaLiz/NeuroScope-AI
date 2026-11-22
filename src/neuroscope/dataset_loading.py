"""Transactional preparation and metadata presentation for loaded datasets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .data.hdf5_source import (
    MatlabV73Source,
    inspect_matlab_v73,
    is_matlab_v73,
    open_matlab_v73,
)
from .data.loaders import DataLoadError, load_array
from .domain import DatasetMetadata, DatasetProfile
from .exploration import DatasetExploration, DatasetSummary, MetadataStatus


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """A validated dataset candidate; ``array`` may be an explicit lazy source."""

    path: Path
    array: npt.NDArray | MatlabV73Source
    profile: DatasetProfile
    exploration: DatasetExploration

    def materialize(self) -> npt.NDArray:
        """Explicitly load the complete dataset into memory when required."""
        if isinstance(self.array, np.ndarray):
            return self.array
        source = self.array
        return source.read_all(channel_axis=self.profile.metadata.channel_axis)

    def close(self) -> None:
        """Release an owned lazy source when the dataset is unloaded."""
        self.exploration.close()


VariableSelector = Callable[[str], str | None]
OrientationSelector = Callable[[tuple[int, ...]], int | None]


def prepare_dataset(
    path: str | Path,
    *,
    delimiter: str | None = None,
    select_mat_variable: VariableSelector | None = None,
    confirm_channel_axis: OrientationSelector | None = None,
) -> PreparedDataset | None:
    """Load and validate a candidate without changing any active application state.

    Selector callbacks return ``None`` when the researcher cancels. In that case
    this function returns ``None`` and does not update or persist anything.
    """
    path = Path(path)
    if path.suffix.lower() == ".mat" and is_matlab_v73(path):
        return _prepare_matlab_v73(
            path,
            select_mat_variable=select_mat_variable,
            confirm_channel_axis=confirm_channel_axis,
        )

    variable_name: str | None = None
    try:
        array = load_array(path, delimiter=delimiter)
    except DataLoadError as exc:
        if path.suffix.lower() != ".mat" or "multiple numeric variables" not in str(exc):
            raise
        if select_mat_variable is None:
            raise
        variable_name = select_mat_variable(str(exc))
        if variable_name is None:
            return None
        variable_name = variable_name.strip()
        if not variable_name:
            raise DataLoadError("Enter a MATLAB numeric variable name or cancel loading.")
        array = load_array(path, variable_name=variable_name)

    channel_axis: int | None = None
    if array.ndim == 2:
        if confirm_channel_axis is None:
            raise ValueError("A 2D dataset requires explicit channel orientation confirmation.")
        channel_axis = confirm_channel_axis(tuple(array.shape))
        if channel_axis is None:
            return None
        if (
            isinstance(channel_axis, bool)
            or not isinstance(channel_axis, int)
            or channel_axis not in (0, 1)
        ):
            raise ValueError("Confirmed channel_axis must be 0 or 1.")

    profile = DatasetProfile.from_array(
        path,
        array,
        variable_name=variable_name,
        channel_axis=channel_axis,
    )
    exploration = DatasetExploration(array, profile)
    return PreparedDataset(path, array, profile, exploration)


def _prepare_matlab_v73(
    path: Path,
    *,
    select_mat_variable: VariableSelector | None,
    confirm_channel_axis: OrientationSelector | None,
) -> PreparedDataset | None:
    """Inspect v7.3 metadata and keep the selected numeric dataset lazy."""
    candidates = inspect_matlab_v73(path)
    if not candidates:
        raise DataLoadError(
            "No supported numeric MATLAB v7.3 datasets were found. "
            "Character arrays, cells, structs, and object references are unsupported."
        )
    if len(candidates) == 1:
        selected = candidates[0]
    else:
        message = (
            f"MAT file contains multiple numeric datasets "
            f"({', '.join(item.path for item in candidates)}); select one explicitly."
        )
        if select_mat_variable is None:
            raise DataLoadError(message)
        selected_name = select_mat_variable(message)
        if selected_name is None:
            return None
        selected = next(
            (item for item in candidates if item.path == selected_name.strip()), None
        )
        if selected is None:
            raise DataLoadError(
                f"MAT dataset {selected_name!r} is not a supported numeric dataset."
            )

    source = open_matlab_v73(path, selected.path)
    try:
        channel_axis: int | None = None
        if len(source.shape) == 2:
            if confirm_channel_axis is None:
                raise ValueError(
                    "A 2D dataset requires explicit channel orientation confirmation."
                )
            channel_axis = confirm_channel_axis(source.shape)
            if channel_axis is None:
                source.close()
                return None
            if (
                isinstance(channel_axis, bool)
                or not isinstance(channel_axis, int)
                or channel_axis not in (0, 1)
            ):
                raise ValueError("Confirmed channel_axis must be 0 or 1.")

        profile = DatasetProfile.from_array(
            path,
            source,
            variable_name=selected.path,
            channel_axis=channel_axis,
        )
        profile = profile.model_copy(
            update={
                "file_format": "matlab-v7.3",
                "dataset_path": selected.path,
                "content_validation": "windowed",
            }
        )
        if source.metadata:
            updates: dict[str, object] = {}
            supported_metadata = {
                key: value
                for key, value in source.metadata.items()
                if key in DatasetMetadata.model_fields
            }
            if supported_metadata:
                metadata = DatasetMetadata.model_validate(
                    {**profile.metadata.model_dump(), **supported_metadata}
                )
                inferred = list(dict.fromkeys(profile.inferred_fields + list(supported_metadata)))
                unknown = [name for name in profile.unknown_fields if name not in supported_metadata]
                updates.update(
                    metadata=metadata,
                    inferred_fields=inferred,
                    unknown_fields=unknown,
                )
            if updates:
                profile = profile.model_copy(update=updates)
        exploration = DatasetExploration(source, profile)
        return PreparedDataset(path, source, profile, exploration)
    except Exception:
        source.close()
        raise


def metadata_panel_text(dataset: PreparedDataset) -> str:
    """Format a dataset summary with field-level evidence labels."""
    summary = dataset.exploration.summary
    lines = [
        f"Dataset: {dataset.path.name}",
        f"File format: {dataset.profile.file_format}",
        f"Original shape: {summary.shape}",
        f"Data type: {summary.dtype}",
        f"Channels: {summary.channel_count}",
        f"Samples: {summary.sample_count}",
    ]
    if dataset.profile.dataset_path is not None:
        lines.append(f"MAT dataset path: {dataset.profile.dataset_path}")
    if dataset.profile.content_validation == "windowed":
        lines.append(
            "Content validation: numeric values are checked as windows are read; "
            "the complete recording has not been scanned."
        )

    if summary.channel_axis is None:
        lines.append("Channel orientation: 1D single-signal vector")
    else:
        axis_status = summary.metadata["channel_axis"].status
        sample_status = summary.metadata["time_axis"].status
        lines.append(
            f"Channel orientation: channels on axis {summary.channel_axis} "
            f"[{axis_status.value.upper()}]; samples on axis {summary.sample_axis} "
            f"[{sample_status.value.upper()}]"
        )

    for field, label in (
        ("sampling_rate_hz", "Sampling rate"),
        ("units", "Units"),
        ("signal_kind", "Signal kind"),
        ("acquisition_origin", "Acquisition origin"),
    ):
        lines.append(f"{label}: {_format_metadata_value(summary, field)}")

    if summary.nominal_duration_s is not None:
        lines.append(f"Nominal duration (N / rate): {summary.nominal_duration_s:g} s [INFERRED]")
        lines.append(
            f"Final sample timestamp: {summary.final_sample_time_s:g} s [INFERRED]"
        )
    notes = summary.metadata["notes"]
    if notes.value is not None:
        lines.append(f"Notes: {_format_metadata_value(summary, 'notes')}")
    return "\n".join(lines)


def plot_axis_labels(summary: DatasetSummary) -> tuple[str, str]:
    """Return labels that never imply an unknown timebase or measurement unit."""
    rate = summary.metadata["sampling_rate_hz"]
    x_label = (
        "Time relative to sample 0 (s)"
        if rate.status is MetadataStatus.CONFIRMED and rate.value is not None
        else "Sample index"
    )
    units = summary.metadata["units"]
    if units.value is None or units.status is MetadataStatus.UNKNOWN:
        y_label = "Value (units unknown)"
    else:
        y_label = f"Value ({units.value}; {units.status.value.upper()})"
    return x_label, y_label


def _format_metadata_value(summary: DatasetSummary, field: str) -> str:
    item = summary.metadata[field]
    if item.status is MetadataStatus.UNKNOWN or item.value is None:
        return "UNKNOWN"
    value = f"{item.value:g}" if isinstance(item.value, (int, float)) else str(item.value)
    return f"{value} [{item.status.value.upper()}]"
