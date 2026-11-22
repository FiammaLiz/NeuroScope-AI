from __future__ import annotations

from contextlib import closing
from pathlib import Path

import numpy as np


class DataLoadError(ValueError):
    """An actionable error while parsing or validating a dataset."""


def load_array(path: str | Path, *, delimiter: str | None = None,
               skiprows: int = 0, header: bool = False,
               variable_name: str | None = None) -> np.ndarray:
    """Load a finite numeric array. Text parsing choices can be explicit."""
    path = Path(path)
    suffix = path.suffix.lower()
    if skiprows < 0:
        raise DataLoadError("skiprows must be a non-negative integer.")
    try:
        if suffix == ".npy":
            if variable_name is not None or header or skiprows or delimiter is not None:
                raise DataLoadError("Text and MAT options cannot be used with .npy files.")
            array = np.load(path, allow_pickle=False)
        elif suffix in {".csv", ".txt"}:
            if variable_name is not None:
                raise DataLoadError("variable_name applies only to .mat files.")
            if suffix == ".txt" and _text_file_has_no_data(path):
                raise DataLoadError("TXT file contains no data (empty or whitespace-only).")
            if delimiter is None:
                if suffix == ".csv": delimiter = ","
                else: raise DataLoadError("Specify delimiter for .txt (for example '\\t').")
            array = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows + int(header))
        elif suffix == ".mat":
            if delimiter is not None or skiprows or header:
                raise DataLoadError("Text parsing options cannot be used with .mat files.")
            array = _load_mat_variable(path, variable_name)
        else:
            raise DataLoadError(f"Unsupported file extension: {suffix or '(none)'}.")
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"Could not load {path.name}: {exc}") from exc
    array = np.asarray(array)
    if array.size == 0:
        raise DataLoadError("The selected data array is empty.")
    if not np.issubdtype(array.dtype, np.number):
        raise DataLoadError(f"The selected data must be numeric (found {array.dtype}).")
    if not np.isfinite(array).all():
        raise DataLoadError("The selected data contains NaN or infinite values.")
    if array.ndim not in (1, 2):
        raise DataLoadError(f"Only 1D or 2D arrays are supported (found {array.ndim}D).")
    return array


def _text_file_has_no_data(path: Path) -> bool:
    """Return whether a text file is empty or contains only whitespace."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            return not any(line.strip() for line in stream)
    except OSError as exc:
        raise DataLoadError(f"Could not read {path.name}: {exc}") from exc


def _load_mat_variable(path: Path, variable_name: str | None) -> np.ndarray:
    """Load one selected numeric classic MAT or v7.3 dataset."""
    import h5py

    from .hdf5_source import inspect_matlab_v73, is_matlab_v73, open_matlab_v73

    if h5py.is_hdf5(path) and not is_matlab_v73(path):
        raise DataLoadError(
            "The .mat file is HDF5-backed but lacks the MATLAB v7.3 MAT-file signature."
        )
    if is_matlab_v73(path):
        candidates = inspect_matlab_v73(path)
        if not candidates:
            raise DataLoadError(
                "No supported numeric MATLAB v7.3 datasets were found. "
                "Character arrays, cells, structs, and object references are unsupported."
            )
        if variable_name is not None:
            selected = next(
                (item for item in candidates if item.path == variable_name), None
            )
            if selected is None:
                available = ", ".join(item.path for item in candidates)
                raise DataLoadError(
                    f"MAT variable '{variable_name}' is not a supported numeric dataset. "
                    f"Available: {available}"
                )
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            available = ", ".join(item.path for item in candidates)
            raise DataLoadError(
                f"MAT file contains multiple numeric variables ({available}); pass variable_name."
            )
        with closing(open_matlab_v73(path, selected.path)) as source:
            array = source.read_all(
                channel_axis=0 if len(selected.shape) == 2 else None
            )
        return _normalize_mat_vector(array)

    try:
        from scipy.io import loadmat
        content = loadmat(path)
        candidates = {k: v for k, v in content.items() if not k.startswith("__")
                      and isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
                      and v.size > 0}
    except (NotImplementedError, ValueError) as exc:
        if isinstance(exc, ValueError) and "Unknown mat file type" not in str(exc) and "7.3" not in str(exc):
            raise
        candidates = None
    if candidates is None:
        raise DataLoadError(
            "Could not read MAT file. Check it is a supported classic MAT or MATLAB v7.3 file."
        )
    if not candidates:
        raise DataLoadError("No supported numeric variable found in MAT file.")
    if variable_name is not None:
        if variable_name not in candidates:
            raise DataLoadError(f"MAT variable '{variable_name}' is not a numeric candidate. Available: {', '.join(candidates)}")
        return _normalize_mat_vector(candidates[variable_name])
    if len(candidates) != 1:
        raise DataLoadError(f"MAT file contains multiple numeric variables ({', '.join(candidates)}); pass variable_name.")
    return _normalize_mat_vector(next(iter(candidates.values())))


def _normalize_mat_vector(array: np.ndarray) -> np.ndarray:
    """Normalize MATLAB row/column vectors while preserving true matrices."""
    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1)
    return array
