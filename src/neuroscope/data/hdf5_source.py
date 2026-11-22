"""Lazy, session-owned access to numeric MATLAB v7.3 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from .loaders import DataLoadError


@dataclass(frozen=True, slots=True)
class MatlabDatasetInfo:
    """Numeric HDF5 dataset metadata in MATLAB logical dimension order."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    attributes: dict[str, Any]
    physical_shape: tuple[int, ...]


def is_matlab_v73(path: str | Path) -> bool:
    """Check the MATLAB v7.3 user-block signature and HDF5 container signature."""
    path = Path(path)
    try:
        with path.open("rb") as stream:
            header = stream.read(128)
        return header.startswith(b"MATLAB 7.3 MAT-file") and bool(h5py.is_hdf5(path))
    except OSError:
        return False


def inspect_matlab_v73(path: str | Path) -> tuple[MatlabDatasetInfo, ...]:
    """List supported numeric datasets without reading their array contents."""
    path = Path(path)
    if not is_matlab_v73(path):
        raise DataLoadError(
            f"{path.name} is not identified as a MATLAB v7.3 HDF5 MAT file."
        )
    try:
        with h5py.File(path, "r") as h5:
            infos: list[MatlabDatasetInfo] = []

            def visit(name: str, item: h5py.Group | h5py.Dataset) -> bool | None:
                if isinstance(item, h5py.Group):
                    matlab_class = _attribute_text(item.attrs.get("MATLAB_class"))
                    if (
                        _under_matlab_refs(name)
                        or matlab_class in {"cell", "struct"}
                        or "MATLAB_sparse" in item.attrs
                    ):
                        return True
                    return None
                if not isinstance(item, h5py.Dataset) or _under_matlab_refs(name):
                    return
                matlab_class = _attribute_text(item.attrs.get("MATLAB_class"))
                if matlab_class in {"char", "cell", "struct", "function_handle", "opaque"}:
                    return
                if not _is_numeric_hdf5_dataset(item):
                    return
                logical_shape = _logical_shape(item.shape)
                if (
                    len(logical_shape) not in (1, 2)
                    or any(size <= 0 for size in logical_shape)
                ):
                    return
                infos.append(
                    MatlabDatasetInfo(
                        path=name,
                        shape=logical_shape,
                        dtype=str(item.dtype),
                        attributes=_supported_attributes(item.attrs),
                        physical_shape=tuple(item.shape),
                    )
                )

            h5.visititems(visit)
            return tuple(infos)
    except OSError as exc:
        raise DataLoadError(
            f"Could not inspect MATLAB v7.3 HDF5 file {path.name}: {exc}"
        ) from exc


def open_matlab_v73(path: str | Path, dataset_path: str) -> MatlabV73Source:
    """Open one supported numeric dataset and retain its HDF5 file handle."""
    return MatlabV73Source(path, dataset_path)


class MatlabV73Source:
    """A lazy logical MATLAB-array view backed by one open HDF5 dataset.

    MATLAB v7.3 reverses dimension order in its HDF5 representation. Reads
    translate logical indices to reversed physical indices and transpose only
    when returning a whole 2D selection. Row/column vectors are exposed as 1D.
    Each read materializes only the requested selection. The source owns its
    file handle and must be closed explicitly.
    """

    def __init__(self, path: str | Path, dataset_path: str) -> None:
        self.path = Path(path)
        self.dataset_path = dataset_path
        self._lock = RLock()
        if not is_matlab_v73(self.path):
            raise DataLoadError(
                f"{self.path.name} is not identified as a MATLAB v7.3 HDF5 MAT file."
            )
        try:
            self._file = h5py.File(self.path, "r")
            item = self._file.get(dataset_path)
            if not isinstance(item, h5py.Dataset):
                raise DataLoadError(f"MATLAB dataset {dataset_path!r} does not exist.")
            info = _info_for_dataset(dataset_path, item)
            if info is None:
                raise DataLoadError(
                    f"MATLAB item {dataset_path!r} is not a supported numeric dataset. "
                    "Cell arrays, structs, character arrays, and object references are unsupported."
                )
            self._dataset = item
            self.info = info
            self.shape = info.shape
            self.dtype = np.dtype(info.dtype)
            self.metadata = info.attributes
            self._closed = False
        except Exception:
            if hasattr(self, "_file"):
                self._file.close()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def read_window(
        self,
        start: int,
        stop: int,
        *,
        channel_axis: int | None = None,
        channel_index: int | None = None,
    ) -> npt.NDArray:
        """Read ``[start, stop)`` and optionally one confirmed-orientation channel."""
        if self._closed:
            raise DataLoadError("MATLAB v7.3 dataset source is closed.")
        sample_count = self.shape[-1] if len(self.shape) == 1 else None
        if len(self.shape) == 2:
            if channel_axis not in (0, 1):
                raise DataLoadError("Confirm channel orientation before reading a 2D MAT dataset.")
            sample_count = self.shape[1 - channel_axis]
        if len(self.shape) == 1:
            if channel_axis is not None or channel_index not in (None, 0):
                raise DataLoadError("A MATLAB vector has one channel and no channel axis.")
        elif channel_index is not None and not 0 <= channel_index < self.shape[channel_axis]:
            raise DataLoadError("Requested channel index is outside the MATLAB dataset.")
        if (
            isinstance(start, bool)
            or isinstance(stop, bool)
            or not isinstance(start, (int, np.integer))
            or not isinstance(stop, (int, np.integer))
            or start < 0
            or stop <= start
            or stop > sample_count
        ):
            raise DataLoadError("Requested sample interval is invalid or outside the dataset.")

        logical_shape_before_vector_normalization = tuple(reversed(self.info.physical_shape))
        with self._lock:
            if self._closed:
                raise DataLoadError("MATLAB v7.3 dataset source is closed.")
            if len(self.shape) == 1:
                result = self._read_vector(int(start), int(stop), logical_shape_before_vector_normalization)
            else:
                sample_axis = 1 - int(channel_axis)
                logical_selection: list[slice | int] = [slice(None), slice(None)]
                logical_selection[sample_axis] = slice(int(start), int(stop))
                if channel_index is not None:
                    logical_selection[int(channel_axis)] = int(channel_index)
                physical_selection = tuple(reversed(logical_selection))
                result = np.asarray(self._dataset[physical_selection])
                if channel_index is None:
                    result = np.transpose(result)
            result = _convert_complex_compound(result, self._dataset.dtype)
            if result.size == 0:
                raise DataLoadError("The requested MATLAB data window is empty.")
            if not np.isfinite(result).all():
                raise DataLoadError("The requested MATLAB data window contains NaN or infinite values.")
            result.setflags(write=False)
            return result

    def read_all(self, *, channel_axis: int | None = None) -> npt.NDArray:
        """Explicitly materialize the selected dataset for compatibility APIs."""
        if len(self.shape) == 2 and channel_axis not in (0, 1):
            raise DataLoadError("Confirm channel orientation before materializing a 2D MAT dataset.")
        sample_count = (
            self.shape[0]
            if len(self.shape) == 1
            else self.shape[1 - int(channel_axis)]
        )
        return self.read_window(
            0,
            sample_count,
            channel_axis=channel_axis,
        )

    def close(self) -> None:
        """Close the owned HDF5 handle; subsequent reads fail clearly."""
        with self._lock:
            if not self._closed:
                self._file.close()
                self._closed = True

    def _read_vector(
        self,
        start: int,
        stop: int,
        logical_shape: tuple[int, ...],
    ) -> npt.NDArray:
        if self._dataset.ndim == 0:
            return np.asarray(self._dataset[()]).reshape(1)[start:stop]
        varying_logical_axis = next(
            (axis for axis, size in enumerate(logical_shape) if size > 1), None
        )
        if varying_logical_axis is None:
            return np.asarray(self._dataset[tuple(0 for _ in self._dataset.shape)]).reshape(1)[start:stop]
        physical_axis = self._dataset.ndim - 1 - varying_logical_axis
        selection: list[slice | int] = [0] * self._dataset.ndim
        selection[physical_axis] = slice(start, stop)
        return np.asarray(self._dataset[tuple(selection)]).reshape(-1)

def _info_for_dataset(name: str, dataset: h5py.Dataset) -> MatlabDatasetInfo | None:
    matlab_class = _attribute_text(dataset.attrs.get("MATLAB_class"))
    if matlab_class in {"char", "cell", "struct", "function_handle", "opaque"}:
        return None
    if not _is_numeric_hdf5_dataset(dataset):
        return None
    shape = _logical_shape(dataset.shape)
    if len(shape) not in (1, 2) or any(size <= 0 for size in shape):
        return None
    return MatlabDatasetInfo(
        path=name,
        shape=shape,
        dtype=str(_logical_dtype(dataset.dtype)),
        attributes=_supported_attributes(dataset.attrs),
        physical_shape=tuple(dataset.shape),
    )


def _logical_shape(physical_shape: tuple[int, ...]) -> tuple[int, ...]:
    logical = tuple(reversed(physical_shape)) or (1,)
    if len(logical) == 2 and 1 in logical:
        return (int(np.prod(logical)),)
    return logical


def _is_numeric_hdf5_dataset(dataset: h5py.Dataset) -> bool:
    dtype = dataset.dtype
    if dtype.hasobject:
        return False
    if dtype.kind in "iufc":
        return True
    return (
        dtype.fields is not None
        and set(dtype.fields) == {"real", "imag"}
        and all(np.issubdtype(dtype.fields[name][0], np.number) for name in ("real", "imag"))
    )


def _logical_dtype(dtype: np.dtype) -> np.dtype:
    if dtype.fields is None or set(dtype.fields) != {"real", "imag"}:
        return dtype
    component_dtype = np.result_type(dtype.fields["real"][0], dtype.fields["imag"][0])
    return np.dtype(np.complex64 if component_dtype.itemsize <= 4 else np.complex128)


def _convert_complex_compound(values: npt.NDArray, physical_dtype: np.dtype) -> npt.NDArray:
    if physical_dtype.fields is None or set(physical_dtype.fields) != {"real", "imag"}:
        return values
    return values["real"] + 1j * values["imag"]


def _under_matlab_refs(name: str) -> bool:
    return any(part == "#refs#" for part in name.split("/"))


def _attribute_text(value: object) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _attribute_text(value.item())
    if isinstance(value, str):
        return value.rstrip("\x00")
    return None


def _supported_attributes(attributes: h5py.AttributeManager) -> dict[str, Any]:
    supported: dict[str, Any] = {}
    for key in ("sampling_rate_hz", "units", "signal_kind", "filtering_description"):
        if key not in attributes:
            continue
        value = attributes[key]
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace").rstrip("\x00")
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            supported[key] = value
    return supported
