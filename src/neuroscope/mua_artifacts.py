"""Project-owned MUAtrace artifacts, stored separately from JSON records."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import numpy as np
import numpy.typing as npt

_REFERENCE = re.compile(r"artifact://mua/([0-9a-f-]{36})")
_ARRAY_FILES = {
    "mua": "m.npy",
    "times_so": "t.npy",
    "spike_sample_indices": "i.npy",
    "spike_times_s": "s.npy",
    "spike_amplitudes": "a.npy",
}
_LEGACY_ARRAY_FILES = {
    "mua": "mua.npy",
    "times_so": "times_so.npy",
    "spike_sample_indices": "spike_sample_indices.npy",
    "spike_times_s": "spike_times_s.npy",
    "spike_amplitudes": "spike_amplitudes.npy",
}


def _has_infinite_values(array: npt.NDArray, block_samples: int = 65536) -> bool:
    """Check a one-dimensional array without allocating a full-size boolean mask."""
    for start in range(0, array.size, block_samples):
        if np.isinf(array[start : start + block_samples]).any():
            return True
    return False


def _has_nonfinite_values(array: npt.NDArray, block_samples: int = 65536) -> bool:
    for start in range(0, array.size, block_samples):
        if not np.isfinite(array[start : start + block_samples]).all():
            return True
    return False


class MUAArtifactError(ValueError):
    """An MUA artifact is invalid, unavailable, or cannot be written."""


class MUAArtifactStore:
    """Write-once, project-owned calcMUA arrays and their compact manifest."""

    def __init__(self, root: str | Path, *, create_root: bool = True) -> None:
        self.root = Path(root).resolve()
        if create_root:
            self.root.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        mua: npt.ArrayLike,
        times_so: npt.ArrayLike,
        spike_sample_indices: npt.ArrayLike,
        spike_times_s: npt.ArrayLike,
        spike_amplitudes: npt.ArrayLike,
        manifest: dict[str, object],
    ) -> str:
        arrays = {
            "mua": np.asarray(mua),
            "times_so": np.asarray(times_so),
            "spike_sample_indices": np.asarray(spike_sample_indices),
            "spike_times_s": np.asarray(spike_times_s),
            "spike_amplitudes": np.asarray(spike_amplitudes),
        }
        trace = arrays["mua"]
        if (
            trace.ndim != 1
            or trace.size == 0
            or not np.issubdtype(trace.dtype, np.number)
            or np.issubdtype(trace.dtype, np.complexfloating)
        ):
            raise MUAArtifactError("MUA trace must be a non-empty numeric vector.")
        if _has_infinite_values(trace):
            raise MUAArtifactError("MUA trace may contain NaN for no events, but not infinity.")
        if arrays["times_so"].shape != trace.shape:
            raise MUAArtifactError("times_so must have the same shape as the MUA trace.")
        times_so = arrays["times_so"]
        if (
            not np.issubdtype(times_so.dtype, np.number)
            or np.issubdtype(times_so.dtype, np.complexfloating)
            or _has_infinite_values(times_so)
        ):
            raise MUAArtifactError("times_so has an invalid dtype or contains infinity.")
        indices = arrays["spike_sample_indices"]
        times = arrays["spike_times_s"]
        amplitudes = arrays["spike_amplitudes"]
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise MUAArtifactError("spike sample indices must be an integer vector.")
        if times.shape != indices.shape or amplitudes.shape != indices.shape:
            raise MUAArtifactError("spike arrays must have equal lengths.")
        if any(
            not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.complexfloating)
            for array in (times, amplitudes)
        ):
            raise MUAArtifactError("spike times and amplitudes must be real numeric vectors.")
        if _has_nonfinite_values(times) or _has_nonfinite_values(amplitudes):
            raise MUAArtifactError("spike times and amplitudes must be finite.")
        if any(array.dtype.hasobject for array in arrays.values()):
            raise MUAArtifactError("MUA artifact arrays cannot use object dtype.")

        artifact_id = str(uuid4())
        reference = f"artifact://mua/{artifact_id}"
        # Keep the private staging path short for Windows MAX_PATH environments;
        # the published artifact directory still uses the full UUID identity.
        temporary = Path(tempfile.mkdtemp(prefix=".p-", dir=self.root))
        destination = self.root / artifact_id
        try:
            files: dict[str, dict[str, object]] = {}
            for key, array in arrays.items():
                filename = _ARRAY_FILES[key]
                np.save(temporary / filename, array, allow_pickle=False)
                files[filename] = {"shape": list(array.shape), "dtype": str(array.dtype)}
            payload = {**manifest, "artifact_id": artifact_id, "files": files}
            encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
            (temporary / "manifest.json").write_text(encoded, encoding="utf-8")
            if destination.exists():
                raise MUAArtifactError("MUA artifact identifier collision; refusing overwrite.")
            temporary.rename(destination)
        except (OSError, TypeError, ValueError) as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            if isinstance(exc, MUAArtifactError):
                raise
            raise MUAArtifactError(f"could not write MUA artifact: {exc}") from exc
        return reference

    def write_average(
        self,
        *,
        traces: tuple[npt.NDArray, ...],
        times_so: npt.ArrayLike,
        manifest: dict[str, object],
        block_samples: int = 65536,
    ) -> str:
        """Persist a bounded-memory arithmetic mean as a new write-once artifact."""
        if len(traces) < 2:
            raise MUAArtifactError("an averaged MUAtrace requires at least two source traces.")
        if isinstance(block_samples, bool) or not isinstance(block_samples, int) or block_samples <= 0:
            raise MUAArtifactError("block_samples must be a positive integer.")
        arrays = tuple(np.asarray(trace) for trace in traces)
        length = arrays[0].size
        if length == 0 or any(
            array.ndim != 1
            or array.size != length
            or not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.complexfloating)
            for array in arrays
        ):
            raise MUAArtifactError("source MUA traces must be equal-length real numeric vectors.")
        grid = np.asarray(times_so)
        if grid.shape != (length,) or not np.issubdtype(grid.dtype, np.number):
            raise MUAArtifactError("the averaged MUA time grid must match the source trace length.")
        if _has_nonfinite_values(grid):
            raise MUAArtifactError("the averaged MUA time grid must be finite.")

        with tempfile.NamedTemporaryFile(
            prefix=".av-", suffix=".npy", dir=self.root, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        average = None
        try:
            average = np.lib.format.open_memmap(
                temporary_path, mode="w+", dtype=np.float64, shape=(length,)
            )
            for start in range(0, length, block_samples):
                stop = min(start + block_samples, length)
                accumulator = np.zeros(stop - start, dtype=np.float64)
                for array in arrays:
                    block = array[start:stop]
                    if not np.isfinite(block).all():
                        raise MUAArtifactError("nonfinite source MUA values cannot be averaged.")
                    accumulator += block
                average[start:stop] = accumulator / len(arrays)
            average.flush()
            mmap = getattr(average, "_mmap", None)
            if mmap is not None:
                mmap.close()
            average = None
            mapped_average = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
            try:
                return self.write(
                    mua=mapped_average,
                    times_so=grid,
                    spike_sample_indices=np.empty(0, dtype=np.int64),
                    spike_times_s=np.empty(0, dtype=np.float64),
                    spike_amplitudes=np.empty(0, dtype=np.float64),
                    manifest={**manifest, "artifact_type": "averaged_mua_trace"},
                )
            finally:
                mapped_mmap = getattr(mapped_average, "_mmap", None)
                if mapped_mmap is not None:
                    mapped_mmap.close()
        finally:
            if average is not None:
                mmap = getattr(average, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            temporary_path.unlink(missing_ok=True)

    def open(self, reference: str) -> MUAArtifact:
        artifact_id = self._parse_reference(reference)
        directory = self.root / artifact_id
        path = directory / "manifest.json"
        if not path.is_file():
            raise MUAArtifactError(f"MUA artifact {artifact_id} does not exist.")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MUAArtifactError(f"MUA artifact {artifact_id} has an unreadable manifest.") from exc
        if manifest.get("artifact_id") != artifact_id:
            raise MUAArtifactError("MUA artifact manifest identity does not match.")
        stored_files = set(manifest.get("files", {}))
        if stored_files not in (
            set(_ARRAY_FILES.values()),
            set(_LEGACY_ARRAY_FILES.values()),
        ):
            raise MUAArtifactError("MUA artifact manifest files are incomplete or unsupported.")
        if any(not (directory / filename).is_file() for filename in stored_files):
            raise MUAArtifactError("MUA artifact is missing a data file.")
        return MUAArtifact(directory, manifest)

    def delete(self, reference: str) -> None:
        artifact_id = self._parse_reference(reference)
        directory = self.root / artifact_id
        if not directory.is_dir():
            raise MUAArtifactError(f"MUA artifact {artifact_id} does not exist.")
        shutil.rmtree(directory)

    @staticmethod
    def _parse_reference(reference: str) -> str:
        if not isinstance(reference, str):
            raise TypeError("MUA artifact reference must be a string.")
        match = _REFERENCE.fullmatch(reference)
        if match is None:
            raise MUAArtifactError("invalid MUA artifact reference.")
        return match.group(1)


class MUAArtifact:
    """Closeable, read-only NPY memory maps for a MUAtrace and event outputs."""

    _ARRAYS: ClassVar[dict[str, str]] = {
        **_ARRAY_FILES,
    }

    def __init__(self, directory: Path, manifest: dict[str, object]) -> None:
        self.directory = directory
        self.manifest = manifest
        self._closed = False
        self._arrays: dict[str, npt.NDArray] = {}
        try:
            stored_names = set(manifest["files"])
            name_map = (
                _ARRAY_FILES
                if stored_names == set(_ARRAY_FILES.values())
                else _LEGACY_ARRAY_FILES
            )
            for key, filename in name_map.items():
                array = np.load(directory / filename, mmap_mode="r", allow_pickle=False)
                self._arrays[key] = array
                spec = manifest["files"][filename]
                if list(array.shape) != spec.get("shape") or str(array.dtype) != spec.get("dtype"):
                    raise MUAArtifactError(
                        f"MUA artifact array {filename} disagrees with its manifest."
                    )
        except Exception:
            self.close()
            raise

    def __getattr__(self, name: str) -> npt.NDArray:
        if name in self._ARRAYS:
            if self._closed:
                raise MUAArtifactError("MUA artifact is closed.")
            return self._arrays[name]
        raise AttributeError(name)

    def read_window(self, start: int, stop: int) -> npt.NDArray:
        if self._closed:
            raise MUAArtifactError("MUA artifact is closed.")
        if (
            isinstance(start, bool)
            or isinstance(stop, bool)
            or not isinstance(start, int)
            or not isinstance(stop, int)
            or start < 0
            or stop <= start
            or stop > self._arrays["mua"].size
        ):
            raise MUAArtifactError("MUA artifact window must be a valid non-empty [start, stop) interval.")
        return self._arrays["mua"][start:stop]

    def close(self) -> None:
        if self._closed:
            return
        for array in self._arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._arrays.clear()
        self._closed = True

    def __enter__(self) -> MUAArtifact:  # noqa: PYI034 - supports Python 3.10
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
