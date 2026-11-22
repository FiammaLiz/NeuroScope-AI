"""Project-owned, file-backed waveform artifacts and their lifecycle."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import numpy.typing as npt

_ARTIFACT_REFERENCE = re.compile(r"artifact://waveforms/([0-9a-f-]{36})")
_ISI_ARTIFACT_REFERENCE = re.compile(r"artifact://isi/([0-9a-f-]{36})")
_MEASUREMENT_NAMES = {
    "minimum_amplitude",
    "maximum_amplitude",
    "peak_to_peak_amplitude",
    "minimum_position_from_event",
    "maximum_position_from_event",
}


class WaveformArtifactError(ValueError):
    """A waveform artifact is invalid, unavailable, or cannot be written."""


class ISIArtifactError(ValueError):
    """An ISI table artifact is invalid or unavailable."""


class ISIArtifactStore:
    """Project-owned streaming storage for interval tables."""

    def __init__(self, root: str | Path, *, create_root: bool = True) -> None:
        self.root = Path(root).resolve()
        if create_root:
            self.root.mkdir(parents=True, exist_ok=True)

    def writer(self) -> ISIArtifactWriter:
        return ISIArtifactWriter(self)

    def open(self, reference: str) -> ISIArtifact:
        artifact_id = self._parse_reference(reference)
        directory = self.root / artifact_id
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise ISIArtifactError(f"ISI artifact {artifact_id} does not exist.")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ISIArtifactError(
                f"ISI artifact {artifact_id} has an unreadable manifest."
            ) from exc
        if (
            manifest.get("artifact_id") != artifact_id
            or manifest.get("intervals_file") != "intervals.jsonl"
        ):
            raise ISIArtifactError("ISI artifact manifest identity or table entry is invalid.")
        if not (directory / "intervals.jsonl").is_file():
            raise ISIArtifactError("ISI artifact interval table is missing.")
        return ISIArtifact(directory, manifest)

    def delete(self, reference: str) -> None:
        artifact_id = self._parse_reference(reference)
        directory = self.root / artifact_id
        if not directory.is_dir():
            raise ISIArtifactError(f"ISI artifact {artifact_id} does not exist.")
        shutil.rmtree(directory)

    @staticmethod
    def _parse_reference(reference: str) -> str:
        if not isinstance(reference, str):
            raise TypeError("ISI artifact reference must be a string.")
        match = _ISI_ARTIFACT_REFERENCE.fullmatch(reference)
        if match is None:
            raise ISIArtifactError("invalid ISI artifact reference.")
        return match.group(1)


class ISIArtifactWriter:
    """Transactional JSONL writer; interval rows are not retained as a table in RAM."""

    def __init__(self, store: ISIArtifactStore) -> None:
        self.store = store
        self.artifact_id = str(uuid4())
        self.reference = f"artifact://isi/{self.artifact_id}"
        self._temporary_directory = store.root / f".pending-{self.artifact_id}"
        self._final_directory = store.root / self.artifact_id
        self._temporary_directory.mkdir()
        self._table_path = self._temporary_directory / "intervals.jsonl"
        self._stream = self._table_path.open("w", encoding="utf-8", newline="\n")
        self._row_count = 0
        self._closed = False
        self._committed = False

    def write_row(self, row: dict[str, object]) -> None:
        if self._closed or self._committed:
            raise ISIArtifactError("ISI artifact writer is no longer open.")
        try:
            encoded = json.dumps(
                row, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        except (TypeError, ValueError) as exc:
            raise ISIArtifactError(f"ISI interval row is not JSON-safe: {exc}") from exc
        self._stream.write(encoded + "\n")
        self._row_count += 1

    def commit(self, manifest: dict[str, object]) -> str:
        if self._closed or self._committed:
            raise ISIArtifactError("ISI artifact writer is no longer open.")
        payload = {
            **manifest,
            "artifact_id": self.artifact_id,
            "intervals_file": "intervals.jsonl",
            "interval_count": self._row_count,
        }
        try:
            encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ISIArtifactError(f"ISI artifact manifest is not JSON-safe: {exc}") from exc
        self._stream.flush()
        self._stream.close()
        (self._temporary_directory / "manifest.json").write_text(encoded, encoding="utf-8")
        if self._final_directory.exists():
            raise ISIArtifactError("ISI artifact identifier collision; refusing overwrite.")
        self._temporary_directory.rename(self._final_directory)
        self._committed = True
        self._closed = True
        return self.reference

    def close(self) -> None:
        if self._closed:
            return
        self._stream.close()
        if self._temporary_directory.exists():
            shutil.rmtree(self._temporary_directory)
        self._closed = True

    def __enter__(self) -> ISIArtifactWriter:  # noqa: PYI034 - supports Python 3.10
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        if exc_type is not None or not self._committed:
            self.close()


class ISIArtifact:
    """Explicitly closeable reader for a streaming JSONL interval artifact."""

    def __init__(self, directory: Path, manifest: dict[str, object]) -> None:
        self.directory = directory
        self.manifest = manifest
        self._closed = False

    def iter_rows(self):
        if self._closed:
            raise ISIArtifactError("ISI artifact is closed.")
        with (self.directory / "intervals.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                if self._closed:
                    raise ISIArtifactError("ISI artifact was closed while iterating.")
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ISIArtifactError("ISI artifact contains a malformed row.") from exc

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> ISIArtifact:  # noqa: PYI034 - supports Python 3.10
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class WaveformArtifactStore:
    """Persistent storage rooted in an explicit project artifact directory.

    Artifacts are immutable directories published by atomic rename. Waveform
    matrices and summary vectors are separate ``.npy`` files; execution records
    retain only an ``artifact://`` reference and compact metadata. The project
    owner controls retention by retaining or explicitly deleting the project.
    """

    def __init__(self, root: str | Path, *, create_root: bool = True) -> None:
        self.root = Path(root).resolve()
        if create_root:
            self.root.mkdir(parents=True, exist_ok=True)

    def writer(
        self,
        *,
        shape: tuple[int, int],
        dtype: npt.DTypeLike,
    ) -> WaveformArtifactWriter:
        return WaveformArtifactWriter(self, shape=shape, dtype=dtype)

    def open(self, reference: str) -> WaveformArtifact:
        artifact_id = self._parse_reference(reference)
        directory = self.root / artifact_id
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise WaveformArtifactError(f"waveform artifact {artifact_id} does not exist.")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WaveformArtifactError(
                f"waveform artifact {artifact_id} has an unreadable manifest."
            ) from exc
        if manifest.get("artifact_id") != artifact_id:
            raise WaveformArtifactError("waveform artifact manifest identity does not match.")
        return WaveformArtifact(directory, manifest)

    def delete(self, reference: str) -> None:
        """Explicitly delete one artifact owned by this store."""
        artifact_id = self._parse_reference(reference)
        directory = self.root / artifact_id
        if not directory.is_dir():
            raise WaveformArtifactError(f"waveform artifact {artifact_id} does not exist.")
        shutil.rmtree(directory)

    @staticmethod
    def _parse_reference(reference: str) -> str:
        if not isinstance(reference, str):
            raise TypeError("artifact reference must be a string.")
        match = _ARTIFACT_REFERENCE.fullmatch(reference)
        if match is None:
            raise WaveformArtifactError("invalid waveform artifact reference.")
        return match.group(1)


class WaveformArtifactWriter:
    """Transactional writer for a single waveform matrix and its summaries."""

    def __init__(
        self,
        store: WaveformArtifactStore,
        *,
        shape: tuple[int, int],
        dtype: npt.DTypeLike,
    ) -> None:
        if (
            len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
            or shape[1] <= 0
        ):
            raise ValueError("waveform artifact shape must be (non-negative count, positive length).")
        self.store = store
        self.shape = shape
        self.dtype = np.dtype(dtype)
        if self.dtype.hasobject or not np.issubdtype(self.dtype, np.number):
            raise TypeError("waveform artifact dtype must be numeric and non-object.")
        self.artifact_id = str(uuid4())
        self.reference = f"artifact://waveforms/{self.artifact_id}"
        self._temporary_directory = store.root / f".pending-{self.artifact_id}"
        self._final_directory = store.root / self.artifact_id
        self._temporary_directory.mkdir()
        self._waveform_path = self._temporary_directory / "waveforms.npy"
        self._waveforms: npt.NDArray | None
        if shape[0]:
            self._waveforms = np.lib.format.open_memmap(
                self._waveform_path,
                mode="w+",
                dtype=self.dtype,
                shape=shape,
            )
        else:
            self._waveforms = None
        self._next_row = 0
        self._committed = False
        self._closed = False
        self._measurement_files: dict[str, str] = {}
        self._channel_summary_files: dict[str, dict[str, str]] = {}

    @property
    def waveforms(self) -> npt.NDArray:
        if self._waveforms is not None:
            return self._waveforms
        return np.empty(self.shape, dtype=self.dtype)

    def write_row(self, row_index: int, waveform: npt.NDArray) -> None:
        if self._closed or self._committed:
            raise WaveformArtifactError("waveform artifact writer is no longer open.")
        if row_index != self._next_row or row_index >= self.shape[0]:
            raise WaveformArtifactError("waveform rows must be written once in event order.")
        values = np.asarray(waveform)
        if values.shape != (self.shape[1],) or values.dtype != self.dtype:
            raise WaveformArtifactError("waveform row shape or dtype does not match the artifact.")
        if not np.isfinite(values).all():
            raise WaveformArtifactError("waveform rows must contain only finite values.")
        assert self._waveforms is not None
        self._waveforms[row_index, :] = values
        self._next_row += 1

    def write_measurements(self, measurements: dict[str, npt.NDArray]) -> dict[str, str]:
        if set(measurements) != _MEASUREMENT_NAMES:
            raise WaveformArtifactError("waveform measurement names do not match the supported schema.")
        for name, values in measurements.items():
            array = np.asarray(values)
            if array.shape != (self.shape[0],) or array.dtype.hasobject:
                raise WaveformArtifactError(f"measurement {name!r} has an invalid shape or dtype.")
            if not np.isfinite(array).all():
                raise WaveformArtifactError(f"measurement {name!r} contains non-finite values.")
            filename = f"measurement-{name}.npy"
            np.save(self._temporary_directory / filename, array, allow_pickle=False)
            self._measurement_files[name] = filename
        return dict(self._measurement_files)

    def write_channel_summary(
        self,
        channel_index: int,
        *,
        mean: npt.NDArray,
        standard_deviation: npt.NDArray,
        median: npt.NDArray,
    ) -> dict[str, str]:
        if isinstance(channel_index, bool) or not isinstance(channel_index, int) or channel_index < 0:
            raise ValueError("channel_index must be a non-negative integer.")
        references: dict[str, str] = {}
        for kind, values in (
            ("mean", mean),
            ("standard-deviation", standard_deviation),
            ("median", median),
        ):
            array = np.asarray(values)
            if array.shape != (self.shape[1],) or not np.isfinite(array).all():
                raise WaveformArtifactError(f"channel {kind} summary has an invalid shape or values.")
            filename = f"channel-{channel_index}-{kind}.npy"
            np.save(self._temporary_directory / filename, array, allow_pickle=False)
            references[kind] = filename
        self._channel_summary_files[str(channel_index)] = references
        return dict(references)

    def commit(self, manifest: dict[str, object]) -> str:
        if self._closed or self._committed:
            raise WaveformArtifactError("waveform artifact writer is no longer open.")
        if self._next_row != self.shape[0]:
            raise WaveformArtifactError(
                f"expected {self.shape[0]} waveforms, received {self._next_row}."
            )
        if not self._measurement_files:
            raise WaveformArtifactError("event measurements must be written before commit.")
        if self._waveforms is None:
            np.save(self._waveform_path, np.empty(self.shape, dtype=self.dtype), allow_pickle=False)
        else:
            self._waveforms.flush()
            self._close_waveforms()
        payload = {
            **manifest,
            "artifact_id": self.artifact_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "waveforms": {
                "file": "waveforms.npy",
                "shape": list(self.shape),
                "dtype": str(self.dtype),
            },
            "measurements": self._measurement_files,
            "channel_summaries": self._channel_summary_files,
        }
        manifest_path = self._temporary_directory / "manifest.json"
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        if self._final_directory.exists():
            raise WaveformArtifactError("waveform artifact identifier collision; refusing overwrite.")
        self._temporary_directory.rename(self._final_directory)
        self._committed = True
        self._closed = True
        return self.reference

    def close(self) -> None:
        if self._closed:
            return
        self._close_waveforms()
        if self._temporary_directory.exists():
            shutil.rmtree(self._temporary_directory)
        self._closed = True

    def _close_waveforms(self) -> None:
        if isinstance(self._waveforms, np.memmap):
            self._waveforms.flush()
            mmap = getattr(self._waveforms, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._waveforms = None

    def __enter__(self) -> WaveformArtifactWriter:  # noqa: PYI034 - supports Python 3.10
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        if exc_type is not None or not self._committed:
            self.close()


class WaveformArtifact:
    """Read-only, explicitly closeable access to a saved waveform artifact."""

    def __init__(self, directory: Path, manifest: dict[str, object]) -> None:
        self.directory = directory
        self.manifest = manifest
        wave_info = manifest["waveforms"]
        if not isinstance(wave_info, dict) or wave_info.get("file") != "waveforms.npy":
            raise WaveformArtifactError("waveform artifact has an invalid waveform entry.")
        self.waveforms = np.load(directory / "waveforms.npy", mmap_mode="r", allow_pickle=False)
        self._arrays: list[npt.NDArray] = [self.waveforms]
        self._closed = False
        if list(self.waveforms.shape) != wave_info.get("shape") or str(self.waveforms.dtype) != wave_info.get("dtype"):
            self.close()
            raise WaveformArtifactError("waveform artifact array does not match its manifest.")

    def measurement(self, name: str) -> npt.NDArray:
        if self._closed:
            raise WaveformArtifactError("waveform artifact is closed.")
        if name not in _MEASUREMENT_NAMES:
            raise KeyError(name)
        filename = self.manifest["measurements"].get(name)
        if filename != f"measurement-{name}.npy":
            raise WaveformArtifactError("waveform artifact measurement reference is invalid.")
        return self._load_array(filename)

    def channel_summary(self, channel_index: int) -> dict[str, npt.NDArray]:
        if self._closed:
            raise WaveformArtifactError("waveform artifact is closed.")
        filenames = self.manifest["channel_summaries"].get(str(channel_index))
        if filenames is None:
            raise KeyError(f"no waveform summary for channel {channel_index}.")
        expected = {
            "mean": f"channel-{channel_index}-mean.npy",
            "standard-deviation": f"channel-{channel_index}-standard-deviation.npy",
            "median": f"channel-{channel_index}-median.npy",
        }
        if filenames != expected:
            raise WaveformArtifactError("waveform artifact summary references are invalid.")
        return {name: self._load_array(filename) for name, filename in expected.items()}

    def close(self) -> None:
        if self._closed:
            return
        for array in getattr(self, "_arrays", ()):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._arrays.clear()
        self._closed = True

    def _load_array(self, filename: str) -> npt.NDArray:
        array = np.load(self.directory / filename, mmap_mode="r", allow_pickle=False)
        self._arrays.append(array)
        return array

    def __enter__(self) -> WaveformArtifact:  # noqa: PYI034 - supports Python 3.10
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
