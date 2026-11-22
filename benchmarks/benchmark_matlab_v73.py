"""Reproducible, non-scientific benchmark for lazy MATLAB v7.3 access.

The generated HDF5 fixture emulates MATLAB's reversed dimensions and class
attribute; it is not created by MATLAB and does not validate every v7.3 layout.
Run from the repository root with the project interpreter:

    python benchmarks/benchmark_matlab_v73.py
"""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
import time
import tracemalloc
from pathlib import Path

import h5py
import numpy as np

from neuroscope.data.hdf5_source import inspect_matlab_v73, open_matlab_v73
from neuroscope.domain import DatasetMetadata, DatasetProfile
from neuroscope.exploration import DatasetExploration
from neuroscope.rendering import render_exploration_signal


def _timed(callable_):
    tracemalloc.start()
    start = time.perf_counter()
    value = callable_()
    elapsed = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, elapsed, peak_bytes


def run_benchmark(channels: int, samples: int, window_samples: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="neuroscope-v73-benchmark-") as directory:
        path = Path(directory) / "synthetic_v73.mat"
        chunk_rows = min(65_536, samples)
        create_start = time.perf_counter()
        with h5py.File(path, "w", userblock_size=512) as h5:
            # Physical HDF5 shape is reversed from the logical MATLAB array.
            dataset = h5.create_dataset(
                "recording",
                shape=(samples, channels),
                dtype=np.float32,
                chunks=(chunk_rows, 1),
            )
            dataset.attrs["MATLAB_class"] = np.bytes_("single")
            channel_values = np.arange(channels, dtype=np.float32)
            for start in range(0, samples, chunk_rows):
                stop = min(start + chunk_rows, samples)
                sample_values = np.arange(start, stop, dtype=np.float32)
                dataset[start:stop, :] = (
                    sample_values[:, None] / max(samples, 1)
                    + channel_values[None, :] / max(channels, 1)
                )
        description = b"MATLAB 7.3 MAT-file, Platform: Benchmark fixture"
        with path.open("r+b") as stream:
            stream.write(
                description[:116].ljust(116, b" ") + b"\0" * 8 + b"\x00\x02IM"
            )
        creation_seconds = time.perf_counter() - create_start
        file_size_bytes = path.stat().st_size

        infos, inspection_seconds, inspection_peak = _timed(
            lambda: inspect_matlab_v73(path)
        )
        source, file_open_seconds, file_open_peak = _timed(
            lambda: open_matlab_v73(path, "recording")
        )
        profile = DatasetProfile(
            dataset_id="benchmark-v73",
            name="Synthetic benchmark",
            source_path=str(path),
            file_format="matlab-v7.3",
            dataset_path="recording",
            content_validation="windowed",
            shape=list(source.shape),
            dtype=str(source.dtype),
            metadata=DatasetMetadata(channel_axis=0, time_axis=1),
            confirmed_fields=["channel_axis"],
        )
        exploration = DatasetExploration(source, profile)
        bounded_window = min(window_samples, samples)
        stop = samples // 2
        start = max(0, stop - bounded_window)

        first, first_seconds, first_peak = _timed(
            lambda: source.read_window(
                start,
                stop,
                channel_axis=0,
                channel_index=min(channels - 1, channels // 2),
            )
        )
        repeated, repeated_seconds, repeated_peak = _timed(
            lambda: source.read_window(
                start,
                stop,
                channel_axis=0,
                channel_index=min(channels - 1, channels // 2),
            )
        )
        preview, preview_seconds, preview_peak = _timed(
            lambda: render_exploration_signal(
                exploration,
                channel_index=min(channels - 1, channels // 2),
                start_sample=0,
                stop_sample=samples,
                pixel_width=1200,
                chunk_samples=65_536,
            )
        )
        source.close()
        gc.collect()

        return {
            "fixture": "synthetic HDF5 emulation of MATLAB v7.3 dimensions",
            "shape_logical": list(infos[0].shape),
            "dtype": infos[0].dtype,
            "hdf5_file_bytes": file_size_bytes,
            "fixture_creation_seconds_excluded_from_reads": creation_seconds,
            "metadata_inspection_seconds": inspection_seconds,
            "metadata_inspection_python_peak_bytes": inspection_peak,
            "persistent_file_open_seconds": file_open_seconds,
            "persistent_file_open_python_peak_bytes": file_open_peak,
            "first_window_seconds": first_seconds,
            "first_window_python_peak_bytes": first_peak,
            "first_window_array_bytes": int(first.nbytes),
            "repeated_window_seconds": repeated_seconds,
            "repeated_window_python_peak_bytes": repeated_peak,
            "repeated_window_array_bytes": int(repeated.nbytes),
            "preview_seconds": preview_seconds,
            "preview_python_peak_bytes": preview_peak,
            "preview_output_values": len(preview.values),
            "preview_reduced": preview.reduced,
            "memory_note": (
                "tracemalloc reports Python-traced peak allocations; it does not include "
                "all HDF5 library or operating-system page-cache memory."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--window-samples", type=int, default=4096)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if value <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    print(json.dumps(run_benchmark(args.channels, args.samples, args.window_samples), indent=2))


if __name__ == "__main__":
    main()
