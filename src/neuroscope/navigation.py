"""GUI-independent channel and sample-window navigation state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import ceil, floor, isclose, isfinite
from numbers import Integral

from .exploration import DatasetExploration, SampleWindow


@dataclass(slots=True)
class NavigationState:
    """Selected original channel indices and a half-open visible sample window.

    The initial view contains every channel and the full recording. Explicit
    channel changes do not alter the window. Pan operations preserve window
    width and clamp at recording boundaries.
    """

    sample_count: int
    channel_count: int
    selected_channels: tuple[int, ...] = field(init=False)
    visible_window: SampleWindow = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, Integral):
            raise TypeError("sample_count must be an integer.")
        if isinstance(self.channel_count, bool) or not isinstance(self.channel_count, Integral):
            raise TypeError("channel_count must be an integer.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        if self.channel_count <= 0:
            raise ValueError("channel_count must be positive.")
        self.sample_count = int(self.sample_count)
        self.channel_count = int(self.channel_count)
        self.selected_channels = tuple(range(self.channel_count))
        self.visible_window = SampleWindow(0, self.sample_count)

    @classmethod
    def from_exploration(cls, exploration: DatasetExploration) -> NavigationState:
        summary = exploration.summary
        return cls(summary.sample_count, summary.channel_count)

    def select_channels(self, channels: Iterable[int]) -> None:
        selected = tuple(channels)
        for channel in selected:
            self._validate_channel(channel)
        if len(set(selected)) != len(selected):
            raise ValueError("Channel selection must not contain duplicates.")
        self.selected_channels = selected

    def set_window(self, start: int, stop: int) -> None:
        window = SampleWindow(start, stop)
        if window.stop > self.sample_count:
            raise ValueError("Visible window exceeds the recording length.")
        self.visible_window = window

    def zoom(self, start: int, stop: int) -> None:
        """Set an explicit half-open zoom interval in original sample indices."""
        self.set_window(start, stop)

    def pan(self, delta_samples: int) -> None:
        """Shift the current window, preserving width and clamping at edges."""
        if isinstance(delta_samples, bool) or not isinstance(delta_samples, Integral):
            raise TypeError("delta_samples must be an integer.")
        current = self.visible_window
        width = current.stop - current.start
        start = min(max(current.start + int(delta_samples), 0), self.sample_count - width)
        self.visible_window = SampleWindow(start, start + width)

    def set_from_axis_limits(
        self,
        left: float,
        right: float,
        *,
        sampling_rate_hz: float | None,
    ) -> None:
        """Convert plotted coordinates to a non-empty, clamped sample interval.

        The conversion rounds outward so every sample intersecting the
        requested continuous interval remains visible. ``None`` means the
        plotted coordinates are sample indices; otherwise they are seconds.
        """
        if not isfinite(left) or not isfinite(right):
            raise ValueError("Axis limits must be finite.")
        if right <= left:
            raise ValueError("Axis limits must form a non-empty interval.")
        scale = 1.0
        if sampling_rate_hz is not None:
            if not isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
                raise ValueError("sampling_rate_hz must be finite and positive.")
            scale = sampling_rate_hz
        sample_left = left * scale
        sample_right = right * scale
        nearest_left = round(sample_left)
        nearest_right = round(sample_right)
        if isclose(sample_left, nearest_left, rel_tol=1e-12, abs_tol=1e-9):
            sample_left = float(nearest_left)
        if isclose(sample_right, nearest_right, rel_tol=1e-12, abs_tol=1e-9):
            sample_right = float(nearest_right)
        start = max(0, min(self.sample_count - 1, floor(sample_left)))
        stop = max(start + 1, min(self.sample_count, ceil(sample_right)))
        self.visible_window = SampleWindow(start, stop)

    def reset(self) -> None:
        """Restore the complete recording window."""
        self.visible_window = SampleWindow(0, self.sample_count)

    def _validate_channel(self, channel: int) -> None:
        if isinstance(channel, bool) or not isinstance(channel, Integral):
            raise TypeError("Channel indices must be integers.")
        if not 0 <= channel < self.channel_count:
            raise IndexError(f"Channel index {channel} is outside the dataset.")
