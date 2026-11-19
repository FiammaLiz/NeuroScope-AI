from __future__ import annotations

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


def _load_mat_variable(path: Path, variable_name: str | None) -> np.ndarray:
    """Load one selected numeric classic MAT or v7.3 dataset."""
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
        try:
            import h5py
            with h5py.File(path, "r") as h5:
                candidates = {k: np.asarray(v) for k, v in h5.items()
                              if isinstance(v, h5py.Dataset) and v.dtype.kind in "iuf" and v.size}
        except ImportError as exc:
            raise DataLoadError("MATLAB v7.3 requires h5py; install the project dependencies.") from exc
        except Exception as exc:
            raise DataLoadError("Could not read MAT file. Check it is a supported classic MAT or v7.3 file.") from exc
    if not candidates:
        raise DataLoadError("No supported numeric variable found in MAT file.")
    if variable_name is not None:
        if variable_name not in candidates:
            raise DataLoadError(f"MAT variable '{variable_name}' is not a numeric candidate. Available: {', '.join(candidates)}")
        return candidates[variable_name]
    if len(candidates) != 1:
        raise DataLoadError(f"MAT file contains multiple numeric variables ({', '.join(candidates)}); pass variable_name.")
    return next(iter(candidates.values()))
