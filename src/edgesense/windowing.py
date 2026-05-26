"""Sliding window utilities for multivariate time-series data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for sliding window creation.

    Attributes:
        window_size: Number of time steps per window.
        stride: Step size between consecutive windows.
    """

    window_size: int
    stride: int


@dataclass(frozen=True)
class WindowedDataset:
    """Container holding windowed arrays and optional timestamps.

    Attributes:
        windows: Array shaped (num_windows, window_size, num_features).
        start_times: Start timestamps for each window, if provided.
        end_times: End timestamps for each window, if provided.
        feature_columns: Names of the feature columns.
        config: Windowing configuration.
    """

    windows: np.ndarray
    start_times: pd.Series | None
    end_times: pd.Series | None
    feature_columns: list[str]
    config: WindowConfig


def compute_num_windows(num_rows: int, window_size: int, stride: int) -> int:
    """Compute the number of windows produced by sliding windowing.

    Args:
        num_rows: Number of rows in the time series.
        window_size: Number of rows per window.
        stride: Step size between windows.

    Returns:
        Number of windows that can be extracted.
    """

    _validate_window_params(window_size, stride)
    if num_rows < window_size:
        return 0
    return 1 + (num_rows - window_size) // stride


def build_window_start_indices(num_rows: int, window_size: int, stride: int) -> np.ndarray:
    """Compute the starting indices for each sliding window.

    Args:
        num_rows: Number of rows in the time series.
        window_size: Number of rows per window.
        stride: Step size between windows.

    Returns:
        NumPy array of window start indices.
    """

    num_windows = compute_num_windows(num_rows, window_size, stride)
    return np.arange(num_windows, dtype=int) * stride


def build_window_mask(
    row_mask: pd.Series | np.ndarray,
    window_size: int,
    stride: int,
    require_all_true: bool = True,
) -> np.ndarray:
    """Aggregate a per-row mask into a per-window mask.

    This is used to keep only windows that are fully healthy (or contain any
    anomalous rows) without materializing full windows.

    Args:
        row_mask: Boolean mask per row (True indicates healthy).
        window_size: Number of rows per window.
        stride: Step size between windows.
        require_all_true: If True, keep windows where all rows are True.
            If False, keep windows where any row is True.

    Returns:
        Boolean mask per window.
    """

    _validate_window_params(window_size, stride)
    mask_array = np.asarray(row_mask, dtype=bool)
    if mask_array.ndim != 1:
        raise ValueError("Row mask must be a 1D boolean array.")

    if mask_array.size < window_size:
        return np.array([], dtype=bool)

    windowed_mask = np.lib.stride_tricks.sliding_window_view(mask_array, window_size)[::stride]
    return windowed_mask.all(axis=1) if require_all_true else windowed_mask.any(axis=1)


def create_sliding_windows(
    features: pd.DataFrame | np.ndarray,
    window_size: int,
    stride: int,
    timestamps: pd.Series | None = None,
    window_mask: np.ndarray | None = None,
    copy_windows: bool = False,
) -> WindowedDataset:
    """Create sliding windows from a multivariate feature matrix.

    The resulting window tensor follows (num_windows, window_size, num_features)
    so it can be passed directly to models expecting (Batch, Sequence, Features).

    Args:
        features: Feature matrix shaped (num_rows, num_features).
        window_size: Number of rows per window.
        stride: Step size between windows.
        timestamps: Optional timestamps aligned with rows.
        window_mask: Optional boolean mask to filter windows.
        copy_windows: If True, materialize a copy of the window array.

    Returns:
        WindowedDataset with windows and optional timestamps.
    """

    _validate_window_params(window_size, stride)
    if isinstance(features, pd.DataFrame):
        feature_columns = list(features.columns)
        data = features.values
    else:
        feature_columns = [f"feature_{idx}" for idx in range(np.asarray(features).shape[1])]
        data = np.asarray(features)

    if data.ndim != 2:
        raise ValueError("Features must be a 2D array of shape (num_rows, num_features).")

    num_rows, _ = data.shape
    if num_rows < window_size:
        raise ValueError("Not enough rows to build at least one window.")

    windows_view = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=0)[::stride]
    # Move window length ahead of feature dimension -> (num_windows, window_size, num_features)
    windows_view = np.moveaxis(windows_view, -1, 1)
    if window_mask is not None:
        if window_mask.shape[0] != windows_view.shape[0]:
            raise ValueError("Window mask length does not match number of windows.")
        windows_view = windows_view[window_mask]

    windows = windows_view.copy() if copy_windows else windows_view

    start_times, end_times = _build_window_times(
        timestamps=timestamps,
        num_rows=num_rows,
        window_size=window_size,
        stride=stride,
        window_mask=window_mask,
    )

    return WindowedDataset(
        windows=windows,
        start_times=start_times,
        end_times=end_times,
        feature_columns=feature_columns,
        config=WindowConfig(window_size=window_size, stride=stride),
    )


def iter_sliding_windows(
    features: pd.DataFrame | np.ndarray,
    window_size: int,
    stride: int,
    window_mask: np.ndarray | None = None,
) -> Iterable[np.ndarray]:
    """Yield sliding windows without materializing the full window tensor.

    Args:
        features: Feature matrix shaped (num_rows, num_features).
        window_size: Number of rows per window.
        stride: Step size between windows.
        window_mask: Optional boolean mask to filter windows.

    Yields:
        Windows shaped (window_size, num_features).
    """

    _validate_window_params(window_size, stride)
    data = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)
    if data.ndim != 2:
        raise ValueError("Features must be a 2D array of shape (num_rows, num_features).")

    num_rows = data.shape[0]
    num_windows = compute_num_windows(num_rows, window_size, stride)
    if window_mask is not None and window_mask.shape[0] != num_windows:
        raise ValueError("Window mask length does not match number of windows.")

    window_index = 0
    for start in range(0, num_rows - window_size + 1, stride):
        if window_mask is None or window_mask[window_index]:
            yield data[start : start + window_size]
        window_index += 1


def _build_window_times(
    timestamps: pd.Series | None,
    num_rows: int,
    window_size: int,
    stride: int,
    window_mask: np.ndarray | None,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Build per-window start and end timestamps."""

    if timestamps is None:
        return None, None

    timestamp_values = pd.to_datetime(timestamps, errors="raise")
    timestamp_series = (
        timestamp_values
        if isinstance(timestamp_values, pd.Series)
        else pd.Series(timestamp_values)
    )
    if timestamp_series.shape[0] != num_rows:
        raise ValueError("Timestamps length does not match feature rows.")

    start_indices = build_window_start_indices(num_rows, window_size, stride)
    end_indices = start_indices + window_size - 1
    if window_mask is not None:
        start_indices = start_indices[window_mask]
        end_indices = end_indices[window_mask]

    start_times = pd.Series(timestamp_series.iloc[start_indices].values)
    end_times = pd.Series(timestamp_series.iloc[end_indices].values)
    return start_times, end_times


def _validate_window_params(window_size: int, stride: int) -> None:
    """Validate that window parameters are positive."""

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")
